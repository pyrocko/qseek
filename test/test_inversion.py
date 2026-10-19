from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from qseek.models.layered_model import Layer, LayeredModel
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.search import Search
from qseek.synthetics.synthetics import (
    Receiver,
    SyntheticEvent,
    SyntheticEventCatalog,
    _gaussian_function,
)
from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.tracers import RayTracers
from qseek.tracers.utils import LayeredEarthModel1D
from qseek.triggers import ThresholdTrigger
from qseek.utils import Range
from qseek.waveforms.sds import SDSArchive
from qseek_inversion.inversion import InversionLayered1D
from qseek_inversion.velocity_model import LayeredModelInversion
from qseek_inversion.waveforms.synthetics import SyntheticEvents

KM = 1e3
N_EVENTS = 6

# Two layers over a half space, depth in km, vp, vs in km/s, density in g/cm³
VELOCITY_MODEL = """
0.0 4.0 2.3 2.6
2.0 4.0 2.3 2.6
2.0 5.0 2.9 2.7
6.0 5.0 2.9 2.7
6.0 6.0 3.46 2.8
30.0 6.0 3.46 2.8
"""


def earth_model() -> LayeredEarthModel1D:
    return LayeredEarthModel1D(filename=None, raw_file_data=VELOCITY_MODEL)


@pytest.fixture
def import_rundir(tmp_path: Path) -> Path:
    """A run directory with the search.json of a synthetic network."""
    reference = Location(lat=10.0, lon=10.0, north_shift=500.0, east_shift=-300.0)
    # A grid of stations above the volume resolves the depth of the sources
    grid = np.linspace(-6, 6, 5) * KM
    stations = StationInventory(
        stations=[
            Station(
                network="XX",
                station=f"S{idx:02d}",
                lat=10.0,
                lon=10.0,
                north_shift=reference.north_shift + north,
                east_shift=reference.east_shift + east,
            )
            for idx, (north, east) in enumerate(
                (north, east) for north in grid for east in grid
            )
        ]
    )
    archive = tmp_path / "sds"
    archive.mkdir()
    search = Search(
        data_provider=SDSArchive(archives=[archive]),
        stations=stations,
        octree=Octree(
            location=reference,
            root_node_size=2 * KM,
            n_levels=3,
            east_bounds=Range(-6 * KM, 6 * KM),
            north_bounds=Range(-6 * KM, 6 * KM),
            depth_bounds=Range(0 * KM, 8 * KM),
        ),
        ray_tracers=RayTracers(root=[FastMarchingTracer(velocity_model=earth_model())]),
        trigger=ThresholdTrigger(threshold=0.5),
        ignore_boundary=False,
    )
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    (rundir / "search.json").write_text(search.model_dump_json())
    return rundir


def test_gaussian_peak_on_arrival() -> None:
    data = np.zeros(1000)
    data[500] = 1.0
    peak = np.convolve(data, _gaussian_function(sigma=10), mode="same")
    assert np.argmax(peak) == 500


def test_synthetic_event_arrivals() -> None:
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = SyntheticEvent(origin_time=origin, lat=10.0, lon=10.0)
    assert event.first_arrival() is None
    assert event.last_arrival() is None

    receiver = Receiver(network="XX", station="A", lat=10.0, lon=10.0)
    receiver.add_arrival("P", 1.0)
    receiver.add_arrival("S", 5.0)
    event.receivers[receiver.nsl] = receiver
    assert event.first_arrival() == origin + timedelta(seconds=1.0)
    assert event.last_arrival() == origin + timedelta(seconds=5.0)


def test_synthetic_traces_and_export(tmp_path: Path) -> None:
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    catalog = SyntheticEventCatalog()
    # The second event is far apart, the export has an empty batch in between
    for offset in (timedelta(0), timedelta(minutes=25)):
        event = SyntheticEvent(origin_time=origin + offset, lat=10.0, lon=10.0)
        receiver = Receiver(network="XX", station="A", lat=10.0, lon=10.0)
        receiver.add_arrival("P", 1.0)
        receiver.add_arrival("S", 5.0)
        event.receivers[receiver.nsl] = receiver
        catalog.events.append(event)

    assert catalog.get_time_span() == (
        origin,
        origin + timedelta(minutes=25, seconds=5),
    )

    # The window starts after the origin time, the S arrival is inside
    start = origin + timedelta(seconds=3)
    (trace,) = catalog.get_traces(
        start_time=start,
        end_time=start + timedelta(seconds=10),
        phase="S",
        sampling_rate=100.0,
    )
    peak_time = trace.tmin + np.argmax(trace.ydata) * trace.deltat
    assert peak_time == pytest.approx((origin + timedelta(seconds=5)).timestamp())

    catalog.export_catalog(tmp_path / "catalog")
    for file in ("catalog.csv", "catalog.json", "pyrocko_events.yaml"):
        assert (tmp_path / "catalog" / file).exists()
    assert (tmp_path / "catalog" / "stations.yaml").exists()
    assert list((tmp_path / "catalog" / "traces").glob("*.mseed"))


def test_layered_model_inversion_copy_and_perturb() -> None:
    model = LayeredModel(
        layers=[
            Layer(top_depth=0.0, vp=3000.0, vs=1700.0),
            Layer(top_depth=2000.0, vp=5000.0, vs=2900.0),
        ]
    )
    inversion_model = LayeredModelInversion.from_layered_model(model)
    inversion_model.perturb_velocities(0.1, rng=np.random.default_rng(0))

    # The start model is unchanged
    assert [layer.vp for layer in model.layers] == [3000.0, 5000.0]

    for layer, original in zip(inversion_model.layers, model.layers, strict=True):
        assert layer.vp != original.vp
        assert 0.9 <= layer.vp / original.vp <= 1.1
        assert layer.vpvs == pytest.approx(original.vpvs)

    with pytest.raises(ValueError):
        inversion_model.perturb_velocities(0.0)


def test_inversion_config_roundtrip() -> None:
    config = InversionLayered1D().model_dump_json()
    InversionLayered1D.model_validate_json(
        config.replace('"qseek-rundir"', f'"{Path.cwd()}"')
    )


@pytest.mark.asyncio
async def test_synthetic_closed_loop(import_rundir: Path, tmp_path: Path) -> None:
    """Synthetic events located in their own velocity model are recovered."""
    synthetics = SyntheticEvents(
        ray_tracer=FastMarchingTracer(velocity_model=earth_model()),
        n_events=N_EVENTS,
        sigma_samples=5,
    )
    inversion = InversionLayered1D(
        import_rundir=import_rundir,
        event_selection=synthetics,
    )
    await inversion.prepare(rundir=tmp_path)
    assert (tmp_path / "synthetic_catalog" / "catalog.json").exists()

    true_model = inversion.get_start_model()
    result = await inversion.test_velocity_model(true_model)

    synthetic_events = synthetics.get_catalog().events
    smallest_node = inversion._search.octree.smallest_node_size()
    assert len(result.event) == N_EVENTS
    for synthetic in synthetic_events:
        detection = min(
            result.event,
            key=lambda det: abs((det.time - synthetic.origin_time).total_seconds()),
        )
        assert abs((detection.time - synthetic.origin_time).total_seconds()) < 0.05
        # Without node interpolation, the detection is the center of a smallest node
        assert detection.distance_to(synthetic) < smallest_node
    # The semblance of each phase is 1 at the source, the nodes are off by < 433 m
    n_phases = 2
    assert result.cumulative_semblance > 0.75 * n_phases * N_EVENTS

    # A wrong model focuses the energy worse
    slow_model = inversion.get_start_model()
    for layer in slow_model.layers:
        layer.vp *= 0.85
        layer.vs *= 0.85
    slow_result = await inversion.test_velocity_model(slow_model)
    assert slow_result.cumulative_semblance < result.cumulative_semblance


def scaled_velocity_model(factor: float) -> str:
    """The velocity model with all velocities scaled."""
    lines = []
    for line in VELOCITY_MODEL.strip().splitlines():
        depth, vp, vs, density = line.split()
        lines.append(
            f"{depth} {float(vp) * factor:.3f} {float(vs) * factor:.3f} {density}"
        )
    return "\n".join(lines)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., int]]:
    """Run `qseek-inversion` with arguments, return the exit code."""
    from qseek import console as console_module
    from qseek_inversion.app import qseek_inversion

    monkeypatch.setattr(console_module, "NON_INTERACTIVE", False)
    monkeypatch.setattr(console_module.console, "quiet", False)
    handlers = list(logging.root.handlers)
    level = logging.root.level

    def run(*args: str | Path) -> int:
        monkeypatch.setattr(sys, "argv", ["qseek-inversion", *map(str, args)])
        try:
            qseek_inversion.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    yield run

    logging.disable(logging.NOTSET)
    logging.root.setLevel(level)
    for handler in set(logging.root.handlers) - set(handlers):
        logging.root.removeHandler(handler)
        handler.close()


def report_lines(output: str) -> dict[str, str]:
    pairs = (line.split(": ", 1) for line in output.splitlines() if ": " in line)
    return {pair[0]: pair[1] for pair in pairs}


def test_cli_config_and_modules(
    cli: Callable[..., int],
    import_rundir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli("modules") == 0
    output = capsys.readouterr().out
    for name in ("InversionLayered1D", "EventWaveformsSelection", "SyntheticEvents"):
        assert name in output

    assert cli("modules", "SyntheticEvents") == 0
    assert '"waveforms": "SyntheticEvents"' in capsys.readouterr().out

    assert (
        cli(
            "config",
            "--import-rundir",
            import_rundir,
            "--selection",
            "EventWaveformsSelection",
        )
        == 0
    )
    config = InversionLayered1D.model_validate_json(capsys.readouterr().out)
    assert config.import_rundir == import_rundir
    assert config.event_selection.waveforms == "EventSelection"

    assert cli("--non-interactive", "modules", "Unknown") == 2
    assert report_lines(capsys.readouterr().out)["status"] == "failed"
    assert cli("--non-interactive", "invert", tmp_path / "missing.json") == 2


def test_cli_invert(
    cli: Callable[..., int],
    import_rundir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The inversion moves a slow start model towards the true model."""
    search_file = import_rundir / "search.json"
    search = Search.model_validate_json(search_file.read_text())
    search.ray_tracers = RayTracers(
        root=[
            FastMarchingTracer(
                velocity_model=LayeredEarthModel1D(
                    filename=None, raw_file_data=scaled_velocity_model(0.9)
                )
            )
        ]
    )
    search_file.write_text(search.model_dump_json())

    config_file = tmp_path / "inversion.json"
    inversion = InversionLayered1D(
        import_rundir=import_rundir,
        event_selection=SyntheticEvents(
            ray_tracer=FastMarchingTracer(velocity_model=earth_model()),
            n_events=N_EVENTS,
            sigma_samples=5,
        ),
        n_iterations=20,
    )
    config_file.write_text(inversion.model_dump_json())

    assert cli("--non-interactive", "invert", "--check", config_file) == 0
    check = report_lines(capsys.readouterr().out)
    assert check["status"] == "ok"
    assert check["events"] == str(N_EVENTS)

    assert cli("--non-interactive", "invert", config_file) == 0
    result = report_lines(capsys.readouterr().out)
    assert result["status"] == "finished"
    assert float(result["best_semblance"]) > float(result["start_semblance"])

    rundir = Path(result["rundir"])
    assert rundir == (tmp_path / "inversion").resolve()
    assert (rundir / "inversion.log").read_text()
    iterations = (rundir / "iterations.jsonl").read_text().splitlines()
    assert len(iterations) == 21

    best_model = LayeredModel.model_validate_json(
        (rundir / "best_model.json").read_text()
    )
    start_model = LayeredModel.from_earth_model(
        LayeredEarthModel1D(filename=None, raw_file_data=scaled_velocity_model(0.9))
    )
    true_model = LayeredModel.from_earth_model(earth_model())
    misfit_start = sum(
        abs(start.vp - true.vp)
        for start, true in zip(start_model.layers, true_model.layers, strict=True)
    )
    misfit_best = sum(
        abs(best.vp - true.vp)
        for best, true in zip(best_model.layers, true_model.layers, strict=True)
    )
    assert misfit_best < misfit_start

    assert cli("--non-interactive", "summary", rundir) == 0
    summary = report_lines(capsys.readouterr().out)
    assert summary["status"] == "finished"
    assert summary["trials"] == "20"

    # An existing rundir is kept, unless forced into a backup
    assert cli("--non-interactive", "invert", config_file) == 2
    assert "exists" in report_lines(capsys.readouterr().out)["error"]
    assert cli("--non-interactive", "invert", "--force", config_file) == 0
    assert list(tmp_path.glob("inversion.bak-*"))
