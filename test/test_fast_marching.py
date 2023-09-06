from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from lassie.models.station import Stations
from lassie.octree import Octree
from lassie.tracers.fast_marching.fast_marching import (
    FastMarchingPhaseTracer,
    StationTravelTimeVolume,
)
from lassie.tracers.fast_marching.velocity_models import Constant3DVelocityModel

CONSTANT_VELOCITY = 5000
KM = 1e3


@pytest_asyncio.fixture
async def station_travel_times(
    octree: Octree, stations: Stations
) -> StationTravelTimeVolume:
    octree.surface_elevation = 1 * KM
    model = Constant3DVelocityModel(velocity=CONSTANT_VELOCITY, grid_spacing=100.0)
    model_3d = model.get_model(octree, stations)
    travel_times = await StationTravelTimeVolume.calculate_from_eikonal(
        model_3d, stations.stations[0]
    )
    return travel_times


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
    tmp_path: str,
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

    station_travel_times.interpolate_travel_times(octree)


@pytest.mark.asyncio
async def test_fast_marching_tracer(octree: Octree, stations: Stations):
    tracer = FastMarchingPhaseTracer(
        velocity_model=Constant3DVelocityModel(
            velocity=CONSTANT_VELOCITY, grid_spacing=80.0
        )
    )
    await tracer.prepare(octree, stations)