import asyncio
import random
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import aiohttp
import numpy as np
import pytest
from rich.progress import Progress

from qseek.models.catalog import EventCatalog
from qseek.models.detection import EventDetection
from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.octree import Octree
from qseek.tracers.cake import EarthModel, Timing, TravelTimeTree
from qseek.utils import Range, datetime_now

DATA_DIR = Path(__file__).parent / "data"

DATA_URL = "https://data.pyrocko.org/testing/lassie-v2/"
DATA_FILES = {
    "FORGE_3D_5_large.P.mod.hdr",
    "FORGE_3D_5_large.P.mod.buf",
    "FORGE_3D_5_large.S.mod.hdr",
    "FORGE_3D_5_large.S.mod.buf",
}

KM = 1e3


async def download_test_data() -> None:
    request_files = [
        DATA_DIR / filename
        for filename in DATA_FILES
        if not (DATA_DIR / filename).exists()
    ]

    if not request_files:
        return

    async with aiohttp.ClientSession() as session:
        for file in request_files:
            url = DATA_URL + file.name
            with Progress() as progress:
                async with session.get(url) as response:
                    task = progress.add_task(
                        f"Downloading {url}",
                        total=response.content_length,
                    )
                    with file.open("wb") as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            progress.advance(task, len(chunk))


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture(scope="session")
def travel_time_tree() -> TravelTimeTree:
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
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

    asyncio.run(download_test_data())
    return DATA_DIR


@pytest.fixture(scope="session")
def octree() -> Octree:
    return Octree(
        location=Location(
            lat=10.0,
            lon=10.0,
            elevation=1.0 * KM,
        ),
        root_node_size=2 * KM,
        n_levels=3,
        east_bounds=Range(-10 * KM, 10 * KM),
        north_bounds=Range(-10 * KM, 10 * KM),
        depth_bounds=Range(0 * KM, 10 * KM),
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
            elevation=random.uniform(0, 0.8) * KM,
            depth=random.uniform(0, 0.2) * KM,
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
def detections() -> Generator[EventCatalog, None, None]:
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
        yield EventCatalog(rundir=Path(tmpdir), events=detections)
