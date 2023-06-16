from pathlib import Path

import pytest

DATA_PATH = Path(__file__).parent / "data"


def pytest_addoption(parser) -> None:
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture(scope="session")
def plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")
