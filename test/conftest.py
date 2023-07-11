import random
from pathlib import Path

import pytest

from lassie.models.station import Station, Stations
from lassie.octree import Octree
from lassie.tracers.cake import EarthModel, Timing, TraveltimeTree

DATA_PATH = Path(__file__).parent / "data"

KM = 1e3


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture(scope="session")
def traveltime_tree() -> TraveltimeTree:
    return TraveltimeTree.new(
        earthmodel=EarthModel(),
        distance_bounds=(0 * KM, 15 * KM),
        receiver_depth_bounds=(0 * KM, 0 * KM),
        source_depth_bounds=(0 * KM, 10 * KM),
        spatial_tolerance=100,
        time_tolerance=0.05,
        timing=Timing(definition="P,p"),
    )


@pytest.fixture(scope="session")
def octree() -> Octree:
    return Octree(
        center_lat=10.0,
        center_lon=10.0,
        surface_elevation=0.0,
        size_initial=2 * KM,
        size_limit=500,
        east_bounds=(-10 * KM, 10 * KM),
        north_bounds=(-10 * KM, 10 * KM),
        depth_bounds=(0 * KM, 10 * KM),
        absorbing_boundary=1 * KM,
    )


@pytest.fixture(scope="session")
def stations() -> Stations:
    n_stations = 20
    stations: list[Station] = []
    for i_sta in range(n_stations):
        station = Station(
            network="XX",
            station="STA%02d" % i_sta,
            lat=10.0,
            lon=10.0,
            elevation=random.uniform(0, 1) * KM,
            north_shift=random.uniform(-10, 10) * KM,
            east_shift=random.uniform(-10, 10) * KM,
        )
        stations.append(station)
    return Stations(stations=stations)
