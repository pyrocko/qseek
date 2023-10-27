from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pytest
import pytest_asyncio

from lassie.models.location import Location
from lassie.models.station import Station, Stations
from lassie.octree import Octree
from lassie.tracers.cake import (
    DEFAULT_VELOCITY_MODEL_FILE,
    CakeTracer,
    EarthModel,
    Timing,
)
from lassie.tracers.fast_marching.fast_marching import (
    FastMarchingTracer,
    StationTravelTimeVolume,
)
from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    NonLinLocVelocityModel,
    VelocityModel3D,
    VelocityModelLayered,
)
from lassie.utils import datetime_now

CONSTANT_VELOCITY = 5000
KM = 1e3

NON_LIN_LOC_REFERENCE_LOCATION = Location(
    lat=38.50402147,
    lon=-112.8963897,
    elevation=1650.0,
)


def has_grid2time() -> bool:
    # grid2time is an executeable that is part of the NonLinLoc package
    # https://alomax.free.fr/nlloc/soft6.00.html
    try:
        import subprocess

        subprocess.run(["Grid2Time", "-h"], capture_output=True)
    except FileNotFoundError:
        return False
    return True


def stations_inside(
    model: VelocityModel3D,
    nstations: int = 20,
    depth_range: float | None = None,
    seed: int = 0,
) -> Stations:
    stations = []
    rng = np.random.RandomState(seed)
    for i_sta in range(nstations):
        station = Station(
            network="FM",
            station="STA%02d" % i_sta,
            lat=model.center.lat,
            lon=model.center.lon,
            elevation=model.center.elevation,
            north_shift=model.center.north_shift + rng.uniform(*model.north_bounds),
            east_shift=model.center.east_shift + rng.uniform(*model.east_bounds),
            depth=model.center.depth
            + (
                rng.uniform(depth_range)
                if depth_range is not None
                else rng.uniform(*model.depth_bounds)
            ),
        )
        station = station.shifted_origin()
        stations.append(station)
    return Stations(stations=stations)


def octree_cover(model: VelocityModel3D) -> Octree:
    return Octree(
        reference=model.center,
        size_initial=2 * KM,
        size_limit=500,
        east_bounds=model.east_bounds,
        north_bounds=model.north_bounds,
        depth_bounds=model.depth_bounds,
        absorbing_boundary=0,
    )


@pytest_asyncio.fixture
async def station_travel_times(
    octree: Octree, stations: Stations
) -> StationTravelTimeVolume:
    octree.reference.elevation = 1 * KM
    model = Constant3DVelocityModel(velocity=CONSTANT_VELOCITY, grid_spacing=100.0)
    model_3d = model.get_model(octree)
    return await StationTravelTimeVolume.calculate_from_eikonal(
        model_3d, stations.stations[0]
    )


@pytest.mark.asyncio
async def test_load_save(
    station_travel_times: StationTravelTimeVolume,
    tmp_path: str,
) -> None:
    outfile = Path(tmp_path) / "test_fast_marching_station.3dtt"
    station_travel_times.save(outfile)
    assert outfile.exists()

    station_travel_times2 = StationTravelTimeVolume.load(outfile)
    np.testing.assert_equal(
        station_travel_times.travel_times, station_travel_times2.travel_times
    )


@pytest.mark.asyncio
async def test_travel_times_constant_model(
    station_travel_times: StationTravelTimeVolume,
    octree: Octree,
) -> None:
    eikonal_travel_times = []
    source_distances = []
    for node in octree:
        source = node.as_location()
        eikonal_travel_times.append(
            station_travel_times.interpolate_travel_time(source)
        )
        source_distances.append(station_travel_times.station.distance_to(source))

    eikonal_travel_times = np.array(eikonal_travel_times)
    assert np.any(eikonal_travel_times)

    analytical_travel_times = np.array(source_distances) / CONSTANT_VELOCITY
    nan_travel_times = np.isnan(eikonal_travel_times)

    assert np.any(~nan_travel_times)
    np.testing.assert_almost_equal(
        eikonal_travel_times[~nan_travel_times],
        analytical_travel_times[~nan_travel_times],
        decimal=1,
    )

    eikonal_travel_times = station_travel_times.interpolate_nodes(
        octree, method="cubic"
    )

    nan_travel_times = np.isnan(eikonal_travel_times)
    assert np.any(~nan_travel_times)
    np.testing.assert_almost_equal(
        eikonal_travel_times[~nan_travel_times],
        analytical_travel_times[~nan_travel_times],
        decimal=1,
    )


@pytest.mark.asyncio
async def test_travel_times_cake(
    octree: Octree,
    fixed_stations: Stations,
):
    tracer = FastMarchingTracer(
        phase="fm:P",
        velocity_model=VelocityModelLayered(
            grid_spacing=200.0,
            velocity="vp",
            filename=DEFAULT_VELOCITY_MODEL_FILE,
        ),
    )
    await tracer.prepare(octree, fixed_stations)

    cake_tracer = CakeTracer(
        phases={"cake:P": Timing(definition="P,p")},
        earthmodel=EarthModel(
            filename=DEFAULT_VELOCITY_MODEL_FILE,
        ),
    )
    await cake_tracer.prepare(octree, fixed_stations)

    travel_times_fast_marching = tracer.get_travel_times("fm:P", octree, fixed_stations)
    travel_times_cake = cake_tracer.get_travel_times("cake:P", octree, fixed_stations)

    nan_mask = np.isnan(travel_times_cake)
    travel_times_fast_marching[nan_mask] = np.nan
    np.testing.assert_almost_equal(
        travel_times_fast_marching, travel_times_cake, decimal=1
    )


@pytest.mark.asyncio
async def test_fast_marching_phase_tracer(
    octree: Octree, fixed_stations: Stations
) -> None:
    tracer = FastMarchingTracer(
        phase="fm:P",
        velocity_model=Constant3DVelocityModel(
            velocity=CONSTANT_VELOCITY, grid_spacing=80.0
        ),
    )
    await tracer.prepare(octree, fixed_stations)
    tracer.get_travel_times("fm:P", octree, fixed_stations)


@pytest.mark.asyncio
async def test_non_lin_loc(data_dir: Path, octree: Octree, stations: Stations) -> None:
    header_file = data_dir / "FORGE_3D_5_large.P.mod.hdr"

    tracer = FastMarchingTracer(
        phase="fm:P",
        velocity_model=NonLinLocVelocityModel(
            header_file=header_file,
            reference_location=NON_LIN_LOC_REFERENCE_LOCATION,
        ),
    )
    octree = octree_cover(tracer.velocity_model.get_model(octree))
    stations = stations_inside(tracer.velocity_model.get_model(octree))
    await tracer.prepare(octree, stations)
    source = octree[1].as_location()
    tracer.get_arrivals(
        "fm:P",
        event_time=datetime_now(),
        source=source,
        receivers=list(stations),
    )


@pytest.mark.plot
def test_non_lin_loc_model(
    data_dir: Path,
    octree: Octree,
    stations: Stations,
) -> None:
    import matplotlib.pyplot as plt

    header_file = data_dir / "FORGE_3D_5_large.P.mod.hdr"

    model = NonLinLocVelocityModel(
        header_file=header_file,
        reference_location=NON_LIN_LOC_REFERENCE_LOCATION,
    )
    velocity_model = model.get_model(octree).resample(
        grid_spacing=200.0,
        method="linear",
    )

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = velocity_model.get_meshgrid()
    cmap = ax.scatter(
        coords[0],
        coords[1],
        -coords[2],
        s=np.log(velocity_model.velocity_model.ravel() / KM),
        c=velocity_model.velocity_model.ravel(),
    )
    fig.colorbar(cmap)
    plt.show()


@pytest.mark.plot
@pytest.mark.asyncio
async def test_non_lin_loc_travel_times(data_dir: Path, octree: Octree) -> None:
    import matplotlib.pyplot as plt

    header_file = data_dir / "FORGE_3D_5_large.P.mod.hdr"

    tracer = FastMarchingTracer(
        phase="fm:P",
        velocity_model=NonLinLocVelocityModel(
            reference_location=NON_LIN_LOC_REFERENCE_LOCATION,
            header_file=header_file,
            grid_spacing=100.0,
        ),
    )
    model_3d = tracer.velocity_model.get_model(octree)
    octree = octree_cover(model_3d)
    stations = stations_inside(model_3d, depth_range=500.0)
    await tracer.prepare(octree, stations)

    volume = tracer.get_travel_time_volume(stations.stations[3])

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = volume.get_meshgrid()

    cmap = ax.scatter(
        coords[0],
        coords[1],
        coords[2],
        c=volume.travel_times.ravel(),
        alpha=0.2,
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Depth (m)")
    fig.colorbar(cmap)

    station_offet = volume.station.offset_from(volume.center)
    ax.scatter(*station_offet, s=1000, c="r")
    plt.show()


@pytest.mark.asyncio
def test_non_lin_loc_geometry(
    data_dir: Path,
    tmp_path: Path,
    octree: Octree,
):
    header_file = data_dir / "FORGE_3D_5_large.P.mod.hdr"
    tracer = FastMarchingTracer(
        phase="fm:P",
        velocity_model=NonLinLocVelocityModel(
            reference_location=NON_LIN_LOC_REFERENCE_LOCATION,
            header_file=header_file,
            grid_spacing=100.0,
        ),
    )
    model_3d = tracer.velocity_model.get_model(octree)

    center = model_3d.center.model_copy()

    size_ns = model_3d.north_bounds[1] - model_3d.north_bounds[0]
    size_ew = model_3d.east_bounds[1] - model_3d.east_bounds[0]

    corner_se = center.model_copy()
    corner_se.east_shift -= size_ew / 2
    corner_se.north_shift -= size_ns / 2

    corner_sw = center.model_copy()
    corner_sw.east_shift += size_ew / 2
    corner_sw.north_shift -= size_ns / 2

    corner_ne = center.model_copy()
    corner_ne.east_shift -= size_ew / 2
    corner_ne.north_shift += size_ns / 2

    corner_nw = center.model_copy()
    corner_nw.east_shift += size_ew / 2
    corner_nw.north_shift += size_ns / 2

    locations = {
        "center": center,
        "reference": NON_LIN_LOC_REFERENCE_LOCATION,
        "corner_se": corner_se,
        "corner_sw": corner_sw,
        "corner_ne": corner_ne,
        "corner_nw": corner_nw,
    }

    # write csv
    csv_file = tmp_path / "corners_model.csv"
    with csv_file.open("w") as f:
        f.write("name, lon, lat\n")
        for name, loc in locations.items():
            f.write(f"{name}, {loc.effective_lon}, {loc.effective_lat}\n")


@pytest.mark.asyncio
@pytest.mark.skipif(not has_grid2time(), reason="Grid2Time not installed")
async def test_non_lin_loc_grid2time(
    data_dir: Path,
    tmp_path: Path,
    octree: Octree,
) -> None:
    header_file = data_dir / "FORGE_3D_5_large.P.mod.hdr"
    velocity_model = NonLinLocVelocityModel(
        reference_location=NON_LIN_LOC_REFERENCE_LOCATION,
        header_file=header_file,
        grid_spacing="input",
    )
    velocity_volume = velocity_model.get_model(octree)

    tracer = FastMarchingTracer(phase="fm:P", velocity_model=velocity_model)
    model_3d = tracer.velocity_model.get_model(octree)

    octree = octree_cover(model_3d)
    stations = stations_inside(model_3d, depth_range=0.0)
    await tracer.prepare(octree, stations)

    station = stations.stations[0]
    tt_volume = tracer.get_travel_time_volume(station)

    size_east = tt_volume.east_bounds[1] - tt_volume.east_bounds[0]
    size_north = tt_volume.north_bounds[1] - tt_volume.north_bounds[0]

    corner_sw = tt_volume.center.model_copy()
    corner_sw.east_shift -= size_east * 0.5
    corner_sw.north_shift -= size_north * 0.5

    class Grid2TimeParams(TypedDict):
        messageFlag: Literal[-1, 0, 1, 2, 3, 4]
        randomNumberSeed: int

        latOrig: float
        longOrig: float
        rotAngle: float

        ttimeFileRoot: Path
        outputFileRoot: Path
        waveType: Literal["P", "S"]
        iSwapBytesOnInput: Literal[0, 1]

        gridMode: Literal["GRID3D", "GRID2D"]
        angleMode: Literal["ANGLES_YES", "ANGLES_NO"]

        srcLabel: str
        xSrce: float
        ySrce: float
        zSrce: float
        elev: float
        hs_eps_init: float
        message_flag: Literal[0, 1, 2]

    non_lin_loc_control_file = """
CONTROL {messageFlag} {randomNumberSeed}
TRANS SIMPLE {latOrig} {longOrig} {rotAngle}
GTFILES {ttimeFileRoot} {outputFileRoot} {waveType} {iSwapBytesOnInput}
GTMODE {gridMode} {angleMode}
GTSRCE {srcLabel} XYZ {xSrce:.2f} {ySrce:.2f} {zSrce:.2f} {elev:.2f}
GT_PLFD {hs_eps_init} {message_flag}
"""

    for ista, station in enumerate(stations):
        offset = station.offset_from(NON_LIN_LOC_REFERENCE_LOCATION)

        params: Grid2TimeParams = {
            "messageFlag": 3,
            "randomNumberSeed": 12345,
            "latOrig": NON_LIN_LOC_REFERENCE_LOCATION.effective_lat,
            "longOrig": NON_LIN_LOC_REFERENCE_LOCATION.effective_lon,
            "rotAngle": 0.0,
            "ttimeFileRoot": (header_file.parent / header_file.name.split(".")[0]),
            "outputFileRoot": tmp_path,
            "waveType": "P",
            "iSwapBytesOnInput": 0,
            "gridMode": "GRID3D",
            "angleMode": "ANGLES_NO",
            "srcLabel": f"SRC_{ista:02d}",
            "xSrce": offset[0] / KM,
            "ySrce": offset[1] / KM,
            "zSrce": offset[2] / KM,
            "elev": 0.0,
            "hs_eps_init": 1e-3,
            "message_flag": 1,
        }

        ctrl_file = tmp_path / "test.ctrl"
        ctrl_file.write_text(non_lin_loc_control_file.format(**params))
        proc = await asyncio.create_subprocess_shell(
            f"Grid2Time {ctrl_file}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        await proc.wait()
        assert proc.returncode == 0, f"bad status for Grid2Time \n {stdout} \n {stderr}"

        buf_file = f"*.{params['waveType']}.{params['srcLabel']}.time.buf"

        log_src = f"""
src_velocity {velocity_volume.get_velocity(station):.1f} km/s,
indices: {velocity_volume._get_location_indices(station)}
offset: {offset}
coords lat/lon: {station.effective_lat:.4f}, {station.effective_lon:.4f}
        """

        for buffer in tmp_path.parent.glob(buf_file):
            if buffer.exists():
                break
        else:
            raise FileNotFoundError(
                f"""No buffer file with traveltimes not found
-------- stdout ----------
{stdout.decode()}
-------- stderr ----------
{stderr.decode()}
-------- lassie ----------
{log_src}
"""
            )

        tt_grid2time = np.fromfile(buffer, dtype=np.float32).reshape(
            tt_volume.travel_times.shape
        )
        tt_volume = tracer.get_travel_time_volume(station)

        np.testing.assert_almost_equal(
            tt_volume.travel_times,
            tt_grid2time,
            decimal=1,
            err_msg=f"{stdout.decode()}\n--- lassie ---\n{log_src}",
        )
