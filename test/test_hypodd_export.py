from __future__ import annotations

import asyncio
import itertools
import math
import os
import random
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from qseek.exporters.hypodd import (
    TOP_FIRST_LAYER,
    HypoDD,
    discretize_earthmodel,
    phase_hint,
    round_time,
    station_labels,
)
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
    assert phase_hint("cake:Rayleigh") is None


def synthetic_rundir(rundir: Path, n_events: int = 40, n_stations: int = 15):
    """Write a run directory with picks from constant velocities.

    The detections are shifted by up to 300 m from the true locations, the picks
    are the travel times from the true locations plus 5 ms noise.
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
    archive = rundir.parent / "sds"
    archive.mkdir()
    search = Search(
        project_dir=rundir.parent,
        data_provider=SDSArchive(archive=archive),
        stations=StationInventory(stations=stations),
        octree=Octree(
            location=reference,
            root_node_size=2 * KM,
            n_levels=3,
            east_bounds=Range(-10 * KM, 10 * KM),
            north_bounds=Range(-10 * KM, 10 * KM),
            depth_bounds=Range(0 * KM, 8 * KM),
        ),
        ray_tracers=RayTracers(
            root=[
                ConstantVelocityTracer(phase="constant:P", velocity=VP),
                ConstantVelocityTracer(phase="constant:S", velocity=VS),
            ]
        ),
    )
    rundir.mkdir(parents=True)
    search.write_config(rundir)

    origin = datetime(2024, 5, 20, tzinfo=UTC)
    truth: dict[datetime, Location] = {}
    detections = []
    for i_event in range(n_events):
        true_location = Location(
            lat=reference.lat,
            lon=reference.lon,
            east_shift=rng.gauss(0, 500),
            north_shift=rng.gauss(0, 500),
            depth=3 * KM + rng.gauss(0, 500),
        )
        time = origin + timedelta(minutes=i_event, microseconds=rng.randint(0, 999999))
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
        for station in stations:
            receiver = Receiver.from_station(station)
            receiver.phase_arrivals = {}
            for phase, velocity in (("constant:P", VP), ("constant:S", VS)):
                observed = time + timedelta(
                    seconds=true_location.distance_to(station) / velocity
                    + rng.gauss(0, 0.005)
                )
                modeled = time + timedelta(
                    seconds=detection.distance_to(station) / velocity
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

    catalog = EventCatalog(rundir=rundir)
    catalog.events = detections
    return catalog, truth


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
    ):
        assert (outdir / filename).exists()

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

    hypodd_bin = os.environ.get("HYPODD_BIN")
    if not hypodd_bin and not shutil.which("hypoDD"):
        pytest.skip("hypoDD binaries not available, set HYPODD_BIN")

    env = {**os.environ, "HYPODD_BIN": hypodd_bin or ""}
    await asyncio.to_thread(
        subprocess.run,
        ["./run.sh"],
        cwd=outdir,
        env=env,
        check=True,
        timeout=120,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )

    ids = {}
    for line in (outdir / "event_ids.csv").read_text().splitlines()[1:]:
        event_id, _, time, _ = line.split(",")
        ids[int(event_id)] = datetime.fromisoformat(time)

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

    assert len(errors_relocated) >= 0.9 * len(truth)
    assert np.median(errors_relocated) < 0.3 * np.median(errors_initial)
