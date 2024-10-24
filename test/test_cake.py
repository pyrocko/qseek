from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.octree import Octree
from qseek.tracers.cake import CakeTracer, EarthModel, Timing, TravelTimeTree
from qseek.tracers.constant_velocity import ConstantVelocityTracer
from qseek.utils import Range

KM = 1e3
CONSTANT_VELOCITY = 5 * KM


@pytest.fixture(scope="session")
def small_octree() -> Octree:
    return Octree(
        location=Location(
            lat=10.0,
            lon=10.0,
            elevation=0.2 * KM,
        ),
        root_node_size=0.5 * KM,
        n_levels=3,
        east_bounds=Range(-2 * KM, 2 * KM),
        north_bounds=Range(-2 * KM, 2 * KM),
        depth_bounds=Range(0 * KM, 2 * KM),
    )


@pytest.fixture(scope="session")
def small_stations() -> Stations:
    rng = np.random.default_rng(1232)
    n_stations = 20
    stations: list[Station] = []
    for i_sta in range(n_stations):
        station = Station(
            network="XX",
            station="STA%02d" % i_sta,
            lat=10.0,
            lon=10.0,
            elevation=rng.uniform(0, 0.1) * KM,
            depth=rng.uniform(0, 0.1) * KM,
            north_shift=rng.uniform(-2, 2) * KM,
            east_shift=rng.uniform(-2, 2) * KM,
        )
        stations.append(station)
    return Stations(stations=stations)


def test_sptree_model(travel_time_tree: TravelTimeTree):
    model = travel_time_tree

    with TemporaryDirectory() as d:
        tmp = Path(d)
        file = model.save(tmp)

        model2 = TravelTimeTree.load(file)
        model2._load_sptree()

    source = Location(
        lat=0.0,
        lon=0.0,
        north_shift=1 * KM,
        east_shift=1 * KM,
        depth=5.0 * KM,
    )
    receiver = Location(
        lat=0.0,
        lon=0.0,
        north_shift=0 * KM,
        east_shift=0 * KM,
        depth=0,
    )

    model.get_travel_time(source, receiver)


@pytest.mark.asyncio
async def test_lut(
    travel_time_tree: TravelTimeTree,
    octree: Octree,
    stations: Stations,
) -> None:
    model = travel_time_tree
    await model.init_lut(octree.nodes, stations)

    traveltimes_tree = await model.interpolate_travel_times(octree, stations)
    traveltimes_lut = await model.get_travel_times(octree.nodes, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)

    # Test refilling the LUT
    model._node_lut.clear()
    traveltimes_tree = await model.interpolate_travel_times(octree, stations)
    traveltimes_lut = await model.get_travel_times(octree.nodes, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)
    assert len(model._node_lut) > 0, "did not refill lut"

    stations_selection = stations.model_copy()
    stations_selection.stations = stations_selection.stations[:5]
    traveltimes_tree = await model.interpolate_travel_times(octree, stations_selection)
    traveltimes_lut = await model.get_travel_times(octree.nodes, stations_selection)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)


@pytest.mark.asyncio
async def test_travel_times_constant_velocity(
    small_octree: Octree,
    small_stations: Stations,
):
    octree = small_octree
    stations = small_stations
    octree.n_levels = 3
    cake_tracer = CakeTracer(
        phases={"cake:P": Timing(definition="P,p")},
        earthmodel=EarthModel(
            filename=None,
            raw_file_data=f"""
 -2.0   {CONSTANT_VELOCITY/KM:.1f}    2.0     2.7
 12.0   {CONSTANT_VELOCITY/KM:.1f}    2.0     2.7
""",
        ),
    )
    constant = ConstantVelocityTracer(
        velocity=CONSTANT_VELOCITY,
    )

    await cake_tracer.prepare(octree, stations)

    cake_travel_times = await cake_tracer.get_travel_times("cake:P", octree, stations)
    constant_traveltimes = await constant.get_travel_times(
        "constant:P", octree.nodes, stations
    )

    nan_mask = np.isnan(cake_travel_times)
    logging.warning("percent nan: %.1f", (nan_mask.sum() / nan_mask.size) * 100)

    constant_traveltimes[nan_mask] = np.nan
    np.testing.assert_almost_equal(cake_travel_times, constant_traveltimes, decimal=2)
