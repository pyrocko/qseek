from __future__ import annotations

from collections.abc import Iterator

import pytest

from qseek.station_weights import DensityTaperWeights, DistanceWeights, StationWeights
from qseek.tracers.fast_marching_3d import fast_marching_3d

pytest_plugins = ["qseek.testing"]


@pytest.fixture(params=[DistanceWeights, DensityTaperWeights], ids=lambda c: c.__name__)
def weights_class(request) -> type[StationWeights]:
    """Each station weights model; tests build a new instance from it."""
    return request.param


@pytest.fixture(scope="session", autouse=True)
def fmm_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the travel time volumes of the tests out of the user's cache."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            fast_marching_3d, "FMM_CACHE_DIR", tmp_path_factory.mktemp("fmm-cache")
        )
        yield
