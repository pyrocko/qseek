from datetime import datetime
from pathlib import Path

import pytest

from lassie.config import Config

DATA_PATH = Path(__file__).parent / "data"


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture
def sample_config() -> Config:
    return Config(
        time_span=(
            datetime.fromisoformat("2022-09-24T17:30:00Z"),
            datetime.fromisoformat("2022-09-24T17:45:00Z"),
        ),
        stations_file=DATA_PATH / "6E-stations.yaml",
        waveform_data=[DATA_PATH],
    )
