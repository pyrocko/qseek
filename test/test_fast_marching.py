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
    model: VelocityModel3D, nstations: int = 20, seed: int = 0
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
            depth=model.center.depth + rng.uniform(*model.depth_bounds),
        )
        stations.append(station)
    return Stations(stations=stations)


@pytest_asyncio.fixture
async def station_travel_times(
    octree: Octree, stations: Stations
) -> StationTravelTimeVolume:
    octree.surface_elevation = 1 * KM
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
async def test_load_interpolation(
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
    np.testing.assert_almost_equal(
        eikonal_travel_times[~nan_travel_times],
        analytical_travel_times[~nan_travel_times],
        decimal=1,
    )

    station_travel_times.interpolate_nodes(octree)


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
    stations = stations_inside(tracer.velocity_model.get_model(octree))
    await tracer.prepare(octree, stations)
    source = octree[1].as_location()
    tracer.get_arrivals(
        "fm:P",
        event_time=datetime_now(),
        source=source,
        receivers=list(stations),
    )
