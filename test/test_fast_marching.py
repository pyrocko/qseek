from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from lassie.models.station import Station, Stations
from lassie.octree import Octree
from lassie.tracers.fast_marching.fast_marching import (
    FastMarchingTracer,
    StationTravelTimeVolume,
)
from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    NonLinLocVelocityModel,
    VelocityModel3D,
)
from lassie.utils import datetime_now

CONSTANT_VELOCITY = 5000
KM = 1e3


def stations_inside(
    model: VelocityModel3D,
    nstations: int = 20,
    seed: int = 0,
    depth: float | None = None,
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
            + (depth if depth is not None else rng.uniform(*model.depth_bounds)),
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
async def test_travel_time_interpolation(
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
        velocity_model=NonLinLocVelocityModel(header_file=header_file),
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

    model = NonLinLocVelocityModel(header_file=header_file)
    velocity_model = model.get_model(octree).resample(
        grid_spacing=200.0,
        method="linear",
    )

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = velocity_model.get_meshgrid()
    print(coords[0].shape)
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
            header_file=header_file,
            grid_spacing=100.0,
        ),
    )
    model_3d = tracer.velocity_model.get_model(octree)
    octree = octree_cover(model_3d)
    stations = stations_inside(model_3d, depth=0.0)
    await tracer.prepare(octree, stations)

    volume = tracer.get_travel_time_volume(stations.stations[0])

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = volume.get_meshgrid()
    print(coords[0].shape)

    cmap = ax.scatter(
        coords[0],
        coords[1],
        coords[2],
        c=volume.travel_times.ravel(),
        alpha=0.2,
    )

    station_offet = volume.station.offset_to(volume.center)
    print(station_offet)
    ax.scatter(*station_offet, s=100, c="r")
    fig.colorbar(cmap)
    plt.show()
