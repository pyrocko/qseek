from pathlib import Path

import pytest

from lassie.config import Config
from lassie.models.station import Stations
from lassie.octree import Octree

DATA_PATH = Path(__file__).parent / "data"
km = 1e3


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture
def sample_config() -> Config:
    return Config(
        # squirrel_environment=Path("/data/marius/eifel"),
        # waveform_data=[Path("/data/marius/eifel/**/*.267")],
        # time_span=(
        #     datetime.fromisoformat("2022-09-24T17:30:00Z"),
        #     datetime.fromisoformat("2022-09-24T17:45:00Z"),
        # ),
        # Laacher See
        octree=Octree(
            center_lat=50.41255,
            center_lon=7.26816,
            surface_elevation=0,
            size_initial=2 * km,
            size_limit=0.2 * km,
            east_bounds=(-15 * km, 15 * km),
            north_bounds=(-15 * km, 15 * km),
            depth_bounds=(0 * km, 15 * km),
        ),
        stations=Stations(pyrocko_station_yamls=[DATA_PATH / "6E-stations.yaml"]),
        waveform_data=[DATA_PATH],
    )
