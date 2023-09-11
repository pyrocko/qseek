import random
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import numpy as np
import pytest

from lassie.models.detection import EventDetection, EventDetections
from lassie.models.location import Location
from lassie.models.station import Station, Stations
from lassie.octree import Octree
from lassie.tracers.cake import EarthModel, Timing, TravelTimeTree
from lassie.utils import datetime_now

DATA_DIR = Path(__file__).parent / "data"

KM = 1e3


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture(scope="session")
def traveltime_tree() -> TravelTimeTree:
    return TravelTimeTree.new(
        earthmodel=EarthModel(),
        distance_bounds=(0 * KM, 15 * KM),
        receiver_depth_bounds=(0 * KM, 0 * KM),
        source_depth_bounds=(0 * KM, 10 * KM),
        spatial_tolerance=100,
        time_tolerance=0.05,
        timing=Timing(definition="P,p"),
    )


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def octree() -> Octree:
    return Octree(
        reference=Location(
            lat=10.0,
            lon=10.0,
            elevation=1.0 * KM,
        ),
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


@pytest.fixture(scope="session")
def fixed_stations() -> Stations:
    n_stations = 20
    rng = np.random.RandomState(0)
    stations: list[Station] = []
    for i_sta in range(n_stations):
        station = Station(
            network="FX",
            station="STA%02d" % i_sta,
            lat=10.0,
            lon=10.0,
            elevation=rng.uniform(0, 1) * KM,
            north_shift=rng.uniform(-10, 10) * KM,
            east_shift=rng.uniform(-10, 10) * KM,
        )
        stations.append(station)
    return Stations(stations=stations)


@pytest.fixture(scope="session")
def detections() -> Generator[EventDetections, None, None]:
    n_detections = 2000
    detections: list[EventDetection] = []
    for _ in range(n_detections):
        time = datetime_now() - timedelta(days=random.uniform(0, 365))
        detection = EventDetection(
            lat=10.0,
            lon=10.0,
            east_shift=random.uniform(-10, 10) * KM,
            north_shift=random.uniform(-10, 10) * KM,
            distance_border=1000.0,
            semblance=random.uniform(0, 1),
            time=time,
        )
        detections.append(detection)
    with TemporaryDirectory() as tmpdir:
        yield EventDetections(rundir=Path(tmpdir), detections=detections)
