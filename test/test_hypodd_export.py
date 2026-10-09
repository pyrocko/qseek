from __future__ import annotations

import asyncio
import csv
import itertools
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError
from pyrocko import cake, io, trace
from pyrocko.model import load_events

from qseek.exporters.cross_correlation import CrossCorrelation, PhaseWindow
from qseek.exporters.hypodd import (
    TOP_FIRST_LAYER,
    HypoDD,
    HypoDDSettings,
    IterationSet,
    discretize_earthmodel,
    phase_hint,
    round_time,
    station_labels,
)
from qseek.extras import hypodd_results
from qseek.images.base import ObservedArrival
from qseek.models.catalog import EventCatalog
from qseek.models.detection import (
    EventDetection,
    EventReceivers,
    PhaseDetection,
    Receiver,
)
from qseek.models.layered_model import LayeredModel
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.search import Search
from qseek.tracers.base import ModelledArrival
from qseek.tracers.cake import CakeTracer, Timing
from qseek.tracers.constant_velocity import ConstantVelocityTracer
from qseek.tracers.tracers import RayTracers
from qseek.tracers.utils import LayeredEarthModel1D
from qseek.utils import NSL, Range
from qseek.waveforms.sds import SDSArchive

KM = 1e3
VP = 5000.0
VS = 2900.0

GRADIENT_MODEL_ND = """    -1.0    1.700  0.950   2.8
     0.0    1.700  0.950   2.8
     0.5    1.838  1.028   2.8
     1.0    2.400  1.344   2.8
     3.0    3.895  2.177   2.8
    10.0    5.624  3.144   2.8
    23.0    6.160  3.457   2.8
mantle
    37.0    8.200  4.600   2.8
"""


def vertical_travel_time(tops: list[float], velocities: list[float], depth: float):
    time = 0.0
    bottoms = [*tops[1:], math.inf]
    for top, bottom, velocity in zip(tops, bottoms, velocities, strict=True):
        top = max(top, 0.0)
        if top >= depth:
            break
        time += (min(bottom, depth) - top) / velocity
    return time


def test_discretize_earthmodel() -> None:
    earthmodel = LayeredEarthModel1D(raw_file_data=GRADIENT_MODEL_ND)
    max_depth = 6 * KM
    layers = discretize_earthmodel(earthmodel, max_thickness=500.0, max_depth=max_depth)

    assert layers[0].top == TOP_FIRST_LAYER
    tops = [layer.top for layer in layers]
    assert tops == sorted(tops)
    assert len(layers) <= 30
    # Gradient layers are split down to max_depth, at most 500 m thick
    for top, bottom in itertools.pairwise(tops[1:]):
        if bottom <= max_depth / KM:
            assert bottom - top <= 0.5 + 1e-6
    for layer in layers:
        assert layer.vp_vs == pytest.approx(1.79, abs=0.01)

    # The harmonic mean velocities keep the vertical travel times
    model = LayeredModel.from_earth_model(earthmodel)
    depths = np.linspace(0.0, max_depth, 6001)
    slowness = 1.0 / model.vp_interpolator(depths)
    expected = np.trapezoid(slowness, depths)
    vertical = vertical_travel_time(
        [layer.top * KM for layer in layers],
        [layer.vp * KM for layer in layers],
        max_depth,
    )
    assert vertical == pytest.approx(expected, rel=2e-3)


def test_station_labels() -> None:
    nsls = [
        NSL("XX", "STA01", ""),
        NSL("XX", "LONGNAME", ""),
        NSL("XX", "DUP", ""),
        NSL("YY", "DUP", ""),
        NSL("YY", "DUP", "00"),
    ]
    labels = station_labels(nsls)
    assert labels[nsls[0]] == "STA01"
    assert labels[nsls[1]] == "LONGNAM"
    assert labels[nsls[2]] == "XXDUP"
    assert labels[nsls[3]] == "YYDUP"
    assert len(set(labels.values())) == len(nsls)
    assert all(len(label) <= 7 for label in labels.values())


def test_round_time() -> None:
    time = datetime(2024, 5, 20, 12, 0, 59, 996000, tzinfo=UTC)
    assert round_time(time) == datetime(2024, 5, 20, 12, 1, 0, tzinfo=UTC)
    time = datetime(2024, 5, 20, 12, 0, 1, 123456, tzinfo=UTC)
    assert round_time(time).microsecond == 120000


def test_phase_hint() -> None:
    assert phase_hint("cake:P") == "P"
    assert phase_hint("fm:S") == "S"
    assert phase_hint("constant:Pg") == "P"
    assert phase_hint("cake:p") == "P"
    assert phase_hint("cake:S*") == "S"
    assert phase_hint("cake:Rayleigh") is None
    assert phase_hint("cake:Surface") is None
    assert phase_hint("cake:PmP") is None


def test_settings_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        HypoDD.model_validate_json('{"min_pick": 3}')
    with pytest.raises(ValidationError):
        HypoDD.model_validate_json('{"hypodd": {"solvr": "SVD"}}')
    with pytest.raises(ValidationError):
        HypoDD.model_validate_json('{"hypodd": {"iterations": [{"damp": 50}]}}')
    exporter = HypoDD.model_validate_json(
        '{"max_rms": 0.2, "hypodd": {"solver": "SVD"}}'
    )
    assert exporter.max_rms == 0.2
    assert exporter.hypodd.solver == "SVD"


def cake_travel_time(
    model: cake.LayeredModel, phases: list[cake.PhaseDef], source: Location, receiver
) -> float:
    distance = source.surface_distance_to(receiver)
    arrivals = model.arrivals(
        distances=[distance * cake.m2d],
        phases=phases,
        zstart=source.effective_depth,
        zstop=-receiver.effective_elevation,
    )
    return min(arrival.t for arrival in arrivals)


SAMPLING_INTERVAL = 0.01
# Gains of the P and S wavelets on the channels
GAINS = {"P": {"Z": 1.0, "N": 0.3, "E": 0.2}, "S": {"Z": 0.2, "N": 1.0, "E": 0.8}}


def wavelet(times: np.ndarray, seed: int) -> np.ndarray:
    """A band-limited wavelet of 2 to 12 Hz around 0.15 s after the arrival."""
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(2.0, 12.0, 5)
    phases = rng.uniform(0.0, 2 * np.pi, 5)
    amplitudes = rng.uniform(0.5, 1.0, 5)
    envelope = np.exp(-(((times - 0.15) / 0.12) ** 2))
    oscillation = np.sum(
        amplitudes[:, None]
        * np.sin(2 * np.pi * frequencies[:, None] * times + phases[:, None]),
        axis=0,
    )
    return envelope * oscillation


def write_waveforms(
    archive: Path,
    stations: list[Station],
    arrivals: dict[tuple[int, str], list[tuple[float, float]]],
    start: datetime,
    duration: float,
) -> None:
    """Write synthetic waveforms to an SDS archive.

    Each station and phase has its own wavelet, the same for all events, scaled by
    the event amplitude. The arrivals are (time, amplitude) per station and phase.
    """
    rng = np.random.default_rng(1)
    n_samples = round(duration / SAMPLING_INTERVAL)
    tmin = start.timestamp()
    times = np.arange(n_samples) * SAMPLING_INTERVAL
    for i_station, station in enumerate(stations):
        for i_comp, comp in enumerate("ZNE"):
            data = rng.normal(0.0, 0.01, n_samples)
            for i_phase, phase in enumerate("PS"):
                seed = 1000 * i_station + 10 * i_phase + i_comp
                for time, amplitude in arrivals[(i_station, phase)]:
                    i0 = max(0, int((time - tmin - 1.0) / SAMPLING_INTERVAL))
                    i1 = min(n_samples, int((time - tmin + 1.5) / SAMPLING_INTERVAL))
                    data[i0:i1] += (
                        amplitude
                        * GAINS[phase][comp]
                        * wavelet(times[i0:i1] - (time - tmin), seed)
                    )
            channel = f"HH{comp}"
            tr = trace.Trace(
                network=station.network,
                station=station.station,
                location=station.location,
                channel=channel,
                tmin=tmin,
                deltat=SAMPLING_INTERVAL,
                ydata=data.astype(np.float32),
            )
            path = (
                archive
                / f"{start:%Y}"
                / station.network
                / station.station
                / f"{channel}.D"
                / f"{station.network}.{station.station}.{station.location}."
                f"{channel}.D.{start:%Y}.{start:%j}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            io.save([tr], str(path), format="mseed")


def synthetic_rundir(
    rundir: Path,
    n_events: int = 40,
    n_stations: int = 15,
    model: Literal["constant", "layered"] = "constant",
    waveforms: bool = False,
):
    """Write a run directory with synthetic picks.

    The picks are the travel times from the true locations plus 5 ms noise, for
    constant velocities or for a layered model with gradients. The detections are
    shifted by up to 300 m from the true locations. With `waveforms`, the SDS
    archive holds synthetic waveforms with the true arrivals.
    """
    rng = random.Random(42)
    reference = Location(lat=40.0, lon=14.0)
    stations = [
        Station(
            network="XX",
            station=f"S{i:02d}",
            lat=reference.lat,
            lon=reference.lon,
            east_shift=rng.uniform(-8, 8) * KM,
            north_shift=rng.uniform(-8, 8) * KM,
            elevation=rng.uniform(0, 300),
        )
        for i in range(n_stations)
    ]
    if model == "constant":
        ray_tracers = [
            ConstantVelocityTracer(phase="constant:P", velocity=VP),
            ConstantVelocityTracer(phase="constant:S", velocity=VS),
        ]
    else:
        earthmodel = LayeredEarthModel1D(raw_file_data=GRADIENT_MODEL_ND)
        ray_tracers = [
            CakeTracer(
                earthmodel=earthmodel,
                phases={
                    "cake:P": Timing(definition="P,p"),
                    "cake:S": Timing(definition="S,s"),
                },
            )
        ]
        cake_model = earthmodel.layered_model
        cake_phases = {
            "cake:P": [cake.PhaseDef("P"), cake.PhaseDef("p")],
            "cake:S": [cake.PhaseDef("S"), cake.PhaseDef("s")],
        }

    archive = rundir.parent / "sds"
    archive.mkdir()
    search = Search(
        project_dir=rundir.parent,
        data_provider=SDSArchive(archives=[archive]),
        stations=StationInventory(stations=stations),
        octree=Octree(
            location=reference,
            root_node_size=2 * KM,
            n_levels=3,
            east_bounds=Range(-10 * KM, 10 * KM),
            north_bounds=Range(-10 * KM, 10 * KM),
            depth_bounds=Range(0 * KM, 8 * KM),
        ),
        ray_tracers=RayTracers(root=ray_tracers),
    )
    rundir.mkdir(parents=True)
    search.write_config(rundir)

    def travel_time(phase: str, source: Location, receiver: Station) -> float:
        if model == "constant":
            velocity = VP if phase == "constant:P" else VS
            return source.distance_to(receiver) / velocity
        return cake_travel_time(cake_model, cake_phases[phase], source, receiver)

    phases = [
        tracer_phase
        for tracer in ray_tracers
        for tracer_phase in (tracer.get_available_phases())
    ]
    origin = datetime(2024, 5, 20, tzinfo=UTC)
    truth: dict[datetime, Location] = {}
    detections = []
    arrivals: dict[tuple[int, str], list[tuple[float, float]]] = {
        (i_station, phase): [] for i_station in range(n_stations) for phase in "PS"
    }
    for i_event in range(n_events):
        true_location = Location(
            lat=reference.lat,
            lon=reference.lon,
            east_shift=rng.gauss(0, 500),
            north_shift=rng.gauss(0, 500),
            depth=3 * KM + rng.gauss(0, 500),
        )
        # the waveforms start a minute before the first event
        time = origin + timedelta(
            minutes=i_event + 1, microseconds=rng.randint(0, 999999)
        )
        detection = EventDetection(
            lat=reference.lat,
            lon=reference.lon,
            east_shift=true_location.east_shift + rng.uniform(-300, 300),
            north_shift=true_location.north_shift + rng.uniform(-300, 300),
            depth=true_location.depth + rng.uniform(-300, 300),
            time=time,
            semblance=0.5,
            distance_border=2 * KM,
        )
        receivers = []
        amplitude = rng.uniform(0.5, 2.0)
        for i_station, station in enumerate(stations):
            receiver = Receiver.from_station(station)
            receiver.phase_arrivals = {}
            for phase in phases:
                true_traveltime = travel_time(phase, true_location, station)
                arrivals[(i_station, phase_hint(phase))].append(
                    (time.timestamp() + true_traveltime, amplitude)
                )
                observed = time + timedelta(
                    seconds=true_traveltime + rng.gauss(0, 0.005)
                )
                modeled = time + timedelta(
                    seconds=travel_time(phase, detection, station)
                )
                receiver.add_phase_detection(
                    PhaseDetection(
                        phase=phase,
                        model=ModelledArrival(phase=phase, time=modeled),
                        observed=ObservedArrival(
                            phase=phase, time=observed, detection_value=0.9
                        ),
                    )
                )
            receivers.append(receiver)
        detection.receivers = EventReceivers(
            event_uid=detection.uid, receivers=receivers
        )
        detections.append(detection)
        truth[time] = true_location

    if waveforms:
        write_waveforms(
            archive, stations, arrivals, origin, duration=(n_events + 2) * 60.0
        )

    catalog = EventCatalog(rundir=rundir)
    catalog.events = detections
    return catalog, truth


async def run_hypodd(outdir: Path) -> None:
    hypodd_bin = os.environ.get("HYPODD_BIN")
    if not hypodd_bin and not shutil.which("hypoDD"):
        pytest.skip("hypoDD binaries not available, set HYPODD_BIN")
    await asyncio.to_thread(
        subprocess.run,
        ["./run.sh"],
        cwd=outdir,
        env={**os.environ, "HYPODD_BIN": hypodd_bin or "", "PYTHON": sys.executable},
        check=True,
        timeout=120,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    # run.sh converts the relocations
    n_relocated = len((outdir / "hypoDD.reloc").read_text().splitlines())
    with (outdir / "hypodd_relocations.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == n_relocated
    assert len(load_events(str(outdir / "hypodd_relocations.yaml"))) == n_relocated


def relocation_errors(
    outdir: Path, catalog: EventCatalog, truth: dict[datetime, Location]
) -> tuple[list[float], list[float]]:
    """Distances of the detections and the relocations to the true locations."""
    with (outdir / "event_ids.csv").open(newline="") as f:
        ids = {
            int(row["id"]): datetime.fromisoformat(row["time"])
            for row in csv.DictReader(f)
        }

    errors_initial, errors_relocated = [], []
    detections = {ev.time: ev for ev in catalog}
    for line in (outdir / "hypoDD.reloc").read_text().splitlines():
        values = line.split()
        time = ids[int(values[0])]
        true_location = truth[time]
        relocated = Location(
            lat=float(values[1]), lon=float(values[2]), depth=float(values[3]) * KM
        )
        errors_initial.append(detections[time].distance_to(true_location))
        errors_relocated.append(relocated.distance_to(true_location))
    return errors_initial, errors_relocated


@pytest.mark.asyncio
async def test_hypodd_export(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, truth = synthetic_rundir(rundir)
    await catalog.save()

    outdir = tmp_path / "hypodd"
    exporter = HypoDD(min_picks=10)
    exporter.ph2dt.max_separation = 10 * KM
    await exporter.export(rundir, outdir)

    for filename in (
        "phase.dat",
        "station.dat",
        "ph2dt.inp",
        "hypoDD.inp",
        "event_ids.csv",
        "stations.csv",
        "run.sh",
        "README.md",
        "hypodd_results.py",
    ):
        assert (outdir / filename).exists()

    with (outdir / "event_ids.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(truth)
    detections = {str(ev.uid): ev for ev in catalog}
    for row in rows:
        detection = detections[row["uid"]]
        assert float(row["lat"]) == pytest.approx(detection.effective_lat, abs=1e-6)
        assert float(row["depth"]) == pytest.approx(detection.effective_depth, abs=0.1)
        assert row["magnitude"] == ""

    phase_lines = (outdir / "phase.dat").read_text().splitlines()
    headers = [line for line in phase_lines if line.startswith("#")]
    assert len(headers) == len(truth)
    assert len(phase_lines) - len(headers) == len(truth) * 15 * 2

    control = (outdir / "hypoDD.inp").read_text().splitlines()
    assert control[0] == "hypoDD_2"
    assert all(len(line) < 220 for line in control)
    imod = control[control.index("* IMOD") + 1]
    assert imod == "5"

    station_lines = (outdir / "station.dat").read_text().splitlines()
    assert len(station_lines) == 15

    await run_hypodd(outdir)
    errors_initial, errors_relocated = relocation_errors(outdir, catalog, truth)
    assert len(errors_relocated) >= 0.9 * len(truth)
    assert np.median(errors_relocated) < 0.3 * np.median(errors_initial)


@pytest.mark.asyncio
async def test_hypodd_export_layered(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, truth = synthetic_rundir(rundir, model="layered")
    await catalog.save()

    outdir = tmp_path / "hypodd"
    exporter = HypoDD(min_picks=10)
    exporter.ph2dt.max_separation = 10 * KM
    await exporter.export(rundir, outdir)

    control = (outdir / "hypoDD.inp").read_text().splitlines()
    assert control[control.index("* IMOD") + 1] == "1"
    tops = control[control.index("* TOP:") + 1].split()
    assert float(tops[0]) == TOP_FIRST_LAYER
    assert len(tops) <= 30

    await run_hypodd(outdir)
    errors_initial, errors_relocated = relocation_errors(outdir, catalog, truth)
    assert len(errors_relocated) >= 0.9 * len(truth)
    assert np.median(errors_relocated) < 0.5 * np.median(errors_initial)


def headers(outdir: Path) -> list[list[str]]:
    return [
        line.split()
        for line in (outdir / "phase.dat").read_text().splitlines()
        if line.startswith("#")
    ]


@pytest.mark.asyncio
async def test_hypodd_export_selection(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, _ = synthetic_rundir(rundir, n_events=6)
    events = list(catalog)
    # weak picks: only 4 picks pass min_pick_confidence
    for receiver in events[0].receivers:
        for arrival in receiver.phase_arrivals.values():
            arrival.observed.detection_value = 0.1
    for receiver in events[0].receivers.receivers[:2]:
        for arrival in receiver.phase_arrivals.values():
            arrival.observed.detection_value = 0.9
    # STA/LTA-like confidences above 1
    for receiver in events[1].receivers:
        for arrival in receiver.phase_arrivals.values():
            arrival.observed.detection_value = 4.2
    # picks far from the modeled arrival
    for receiver in events[2].receivers.receivers[:5]:
        arrival = receiver.phase_arrivals["constant:P"]
        arrival.observed.time += timedelta(seconds=3.0)
    # above sea level
    events[3].depth = -100.0
    await catalog.save()

    outdir = tmp_path / "hypodd"
    await HypoDD(min_picks=6).export(rundir, outdir)
    phase_lines = (outdir / "phase.dat").read_text().splitlines()
    header_lines = headers(outdir)
    assert len(header_lines) == 5  # event 0 has too few picks

    blocks: list[list[str]] = []
    for line in phase_lines:
        if line.startswith("#"):
            blocks.append([])
        else:
            blocks[-1].append(line)
    weights = {float(line.split()[2]) for line in blocks[0]}
    assert weights == {1.0}
    assert len(blocks[1]) == 30 - 5
    assert float(header_lines[2][9]) == 0.0
    assert all(float(line.split()[1]) > 0.0 for block in blocks for line in block)

    max_rms = float(np.median([ev.rms for ev in events[1:]]))
    outdir = tmp_path / "hypodd-rms"
    await HypoDD(max_rms=max_rms).export(rundir, outdir)
    expected = sum(1 for ev in events[1:] if ev.rms <= max_rms)
    assert 0 < expected < 5
    assert len(headers(outdir)) == expected

    with pytest.raises(ValueError, match="no detections"):
        await HypoDD(max_rms=1e-9).export(rundir, tmp_path / "hypodd-none")


def test_cli_force_keeps_export_on_error(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, _ = synthetic_rundir(rundir, n_events=10)
    asyncio.run(catalog.save())

    outdir = tmp_path / "hypodd"
    outdir.mkdir()
    (outdir / "marker").touch()
    bad_config = tmp_path / "bad.json"
    bad_config.write_text('{"min_pick": 3}')
    good_config = tmp_path / "good.json"
    good_config.write_text('{"min_picks": 10}')

    def export(config: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "qseek.apps.qseek",
                "export",
                "hypodd",
                str(rundir),
                str(outdir),
                "--force",
                "--config",
                str(config),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )

    assert export(bad_config).returncode != 0
    assert (outdir / "marker").exists()

    assert export(good_config).returncode == 0
    assert not (outdir / "marker").exists()
    assert (outdir / "phase.dat").exists()
    assert not list(tmp_path.glob(".hypodd*"))


RELOC = """\
        2  40.843648   14.135918     1.307       46.6     1941.2     -388.5    102.3    119.4    139.9 2024  5 20  0 40 51.540  0.70     0     0    78    66 -9.000  0.109   1
        1  40.830623   14.146896     1.270      972.6      494.8     -425.9     73.2     83.6     73.3 2024  5 20  0 17 59.995  0.00     0     0    83   103 -9.000  0.078   1
"""


def test_hypodd_results(tmp_path: Path) -> None:
    (tmp_path / "hypoDD.reloc").write_text(RELOC)
    (tmp_path / "event_ids.csv").write_text(
        "id,uid,time,hypodd_time,lat,lon,depth,magnitude,magnitude_type\n"
        "1,uid-1,2024-05-20T00:17:59.958485+00:00,2024-05-20T00:17:59.960000+00:00,"
        "40.830000,14.146000,1500.0,,\n"
        "2,uid-2,2024-05-20T00:40:51.558485+00:00,2024-05-20T00:40:51.560000+00:00,"
        "40.843246,14.146983,2447.8,0.703,ML-campi-flegrei\n"
        "3,uid-3,2024-05-20T01:00:00+00:00,2024-05-20T01:00:00+00:00,"
        "40.8,14.1,1000.0,,\n"
    )
    relocations = hypodd_results.convert(tmp_path)
    assert [r.hypodd_id for r in relocations] == [1, 2]

    with (tmp_path / "hypodd_relocations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(hypodd_results.CSV_COLUMNS)
    first, second = rows
    # ISO 8601, the seconds roll over into the next minute
    assert first["time"] == "2024-05-20T00:17:59.995Z"
    assert first["uid"] == "uid-1"
    assert first["depth"] == "1270.0"
    assert first["magnitude"] == ""
    assert first["rms_cc"] == ""
    assert first["rms_ct"] == "0.0780"
    assert first["WKT_geom"] == "POINT Z(14.146896 40.830623 -1270.0)"
    assert float(first["shift_depth"]) == pytest.approx(-230.0)
    assert float(first["shift_time"]) == pytest.approx(0.037, abs=1e-3)
    assert float(first["shift_north"]) == pytest.approx(
        math.radians(0.000623) * 6371e3, abs=0.1
    )
    assert second["magnitude"] == "0.70"
    assert second["magnitude_type"] == "ML-campi-flegrei"
    assert second["n_ct_p"] == "78"

    events = load_events(str(tmp_path / "hypodd_relocations.yaml"))
    assert [ev.name for ev in events] == [first["time"], second["time"]]
    assert events[0].depth == pytest.approx(1270.0)
    assert events[0].magnitude is None
    assert events[1].magnitude == pytest.approx(0.703)
    assert events[1].extras["qseek_uid"] == "uid-2"
    assert events[1].extras["hypodd_id"] == 2


def test_hypodd_results_old_event_ids(tmp_path: Path) -> None:
    """Exports of the first version list only UID and time in event_ids.csv."""
    (tmp_path / "hypoDD.reloc").write_text(RELOC)
    (tmp_path / "event_ids.csv").write_text(
        "id,uid,time,hypodd_time\n"
        "1,uid-1,2024-05-20T00:17:59.958485+00:00,2024-05-20T00:17:59.960000+00:00\n"
        "2,uid-2,2024-05-20T00:40:51.558485+00:00,2024-05-20T00:40:51.560000+00:00\n"
    )
    hypodd_results.convert(tmp_path)
    with (tmp_path / "hypodd_relocations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["magnitude"] == "0.70"
    assert rows[1]["shift_horizontal"] == ""
    assert rows[1]["qseek_time"] == "2024-05-20T00:40:51.558Z"


def cc_errors(
    outdir: Path, catalog: EventCatalog, truth: dict[datetime, Location]
) -> dict[str, list[float]]:
    """Errors of the differential times in dt.cc to the true travel times."""
    with (outdir / "event_ids.csv").open(newline="") as f:
        ids = {
            int(row["id"]): (
                datetime.fromisoformat(row["time"]),
                datetime.fromisoformat(row["hypodd_time"]),
            )
            for row in csv.DictReader(f)
        }
    receivers = {rcv.nsl.pretty: rcv for rcv in next(iter(catalog)).receivers}
    with (outdir / "stations.csv").open(newline="") as f:
        stations = {row["label"]: receivers[row["nsl"]] for row in csv.DictReader(f)}

    def traveltime(event_id: int, label: str, phase: str) -> float:
        time, hypodd_time = ids[event_id]
        velocity = VP if phase == "P" else VS
        distance = truth[time].distance_to(stations[label])
        return (time - hypodd_time).total_seconds() + distance / velocity

    errors: dict[str, list[float]] = {"P": [], "S": []}
    for line in (outdir / "dt.cc").read_text().splitlines():
        values = line.split()
        if values[0] == "#":
            id_1, id_2 = int(values[1]), int(values[2])
            assert float(values[3]) == 0.0
            continue
        label, dt, weight, phase = values
        assert 0.0 < float(weight) <= 1.0
        expected = traveltime(id_1, label, phase) - traveltime(id_2, label, phase)
        errors[phase].append(float(dt) - expected)
    return errors


@pytest.mark.asyncio
async def test_hypodd_export_cc(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, truth = synthetic_rundir(rundir, waveforms=True)
    # weak picks at three stations: correlated around the modeled arrivals
    first = next(iter(catalog))
    for receiver in first.receivers.receivers[:3]:
        for arrival in receiver.phase_arrivals.values():
            arrival.observed.detection_value = 0.1
    await catalog.save()

    outdir_ct = tmp_path / "hypodd-ct"
    exporter = HypoDD(min_picks=10)
    exporter.ph2dt.max_separation = 10 * KM
    await exporter.export(rundir, outdir_ct)

    outdir = tmp_path / "hypodd-cc"
    exporter = HypoDD(min_picks=10, cross_correlation=CrossCorrelation())
    exporter.ph2dt.max_separation = 10 * KM
    await exporter.export(rundir, outdir)

    control = (outdir / "hypoDD.inp").read_text().splitlines()
    assert control[control.index("* IDAT IPHA DIST") + 1].split()[0] == "3"
    assert "dt.cc" in control
    assert "station.dat" in control
    weighting = control[
        control.index("* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP") + 1 :
    ]
    assert len(weighting[0].split()) == 10

    errors = cc_errors(outdir, catalog, truth)
    assert len(errors["P"]) > 1000
    assert len(errors["S"]) > 1000
    for phase_errors in errors.values():
        assert np.median(np.abs(phase_errors)) < 0.001
        assert np.max(np.abs(phase_errors)) < 0.005
    # the weak picks of the first event are not in phase.dat, but in dt.cc
    blocks = (outdir / "dt.cc").read_text().split("# 1 ")[1:]
    labels = {line.split()[0] for block in blocks for line in block.splitlines()[1:]}
    weak = {rcv.station for rcv in first.receivers.receivers[:3]}
    assert weak <= labels

    await run_hypodd(outdir_ct)
    await run_hypodd(outdir)
    _, errors_ct = relocation_errors(outdir_ct, catalog, truth)
    errors_initial, errors_cc = relocation_errors(outdir, catalog, truth)
    assert len(errors_cc) >= 0.9 * len(truth)
    assert np.median(errors_cc) < 0.2 * np.median(errors_initial)
    assert np.median(errors_cc) < np.median(errors_ct)


def test_cc_iterations() -> None:
    exporter = HypoDD(cross_correlation=CrossCorrelation())
    lines = [it.as_line() for it in exporter.hypodd.iterations]
    # Table 1 of the HypoDD user guide
    assert lines == [
        "5 0.01 0.01 -999 -999 1 0.5 -999 -999 80",
        "5 0.01 0.01 -999 -999 1 0.5 6 4 80",
        "5 1 0.5 -999 2 0.01 0.005 6 4 80",
        "5 1 0.5 6 2 0.01 0.005 6 4 80",
        "5 1 0.5 6 0.5 0.01 0.005 6 4 80",
    ]
    # the settings of the export reproduce the export
    copy = HypoDD.model_validate_json(exporter.model_dump_json())
    assert copy.hypodd.iterations == exporter.hypodd.iterations

    iterations = [IterationSet(weight_cc_p=1.0, weight_cc_s=1.0)]
    exporter = HypoDD.model_validate(
        {
            "cross_correlation": {"bandpass": [2.0, 10.0]},
            "hypodd": {"iterations": [it.model_dump() for it in iterations]},
        }
    )
    assert exporter.hypodd.iterations == iterations
    assert HypoDD().hypodd.iterations[0].as_line().startswith("5 -999 -999 -999 -999")
    with pytest.raises(ValidationError):
        CrossCorrelation(bandpass=(10.0, 2.0))


def test_cc_iterations_after_construction() -> None:
    settings = HypoDDSettings()
    exporter = HypoDD(hypodd=settings, cross_correlation=CrossCorrelation())
    assert len(exporter.hypodd.iterations) == 5
    # the shared settings of hypoDD stay as they are
    assert len(settings.iterations) == 3

    exporter = HypoDD()
    exporter.cross_correlation = CrossCorrelation()
    exporter.set_cc_iterations()
    assert all(it.uses_cc() for it in exporter.hypodd.iterations)


@pytest.mark.asyncio
async def test_hypodd_export_cc_without_times(tmp_path: Path) -> None:
    rundir = tmp_path / "run"
    catalog, _ = synthetic_rundir(rundir, n_events=6, waveforms=True)
    await catalog.save()

    # no channel has the orientation X: no waveforms to correlate
    exporter = HypoDD(min_picks=10)
    exporter.cross_correlation = CrossCorrelation(
        window_p=PhaseWindow(
            seconds_before=0.1, seconds_after=0.5, max_lag=0.2, components="X"
        ),
        window_s=PhaseWindow(
            seconds_before=0.2, seconds_after=1.0, max_lag=0.3, components="X"
        ),
    )
    outdir = tmp_path / "hypodd"
    await exporter.export(rundir, outdir)

    assert not (outdir / "dt.cc").exists()
    control = (outdir / "hypoDD.inp").read_text().splitlines()
    assert control[control.index("* IDAT IPHA DIST") + 1].split()[0] == "2"
    assert "station.sel" in control
    # export() switched to the weighting of the cross-correlation data
    info = HypoDD.model_validate_json((outdir / "export_info.json").read_text())
    assert len(info.hypodd.iterations) == 5
