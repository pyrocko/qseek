from __future__ import annotations

import pytest

from qseek.station_weights import DensityTaperWeights, DistanceWeights, StationWeights

pytest_plugins = ["qseek.testing"]


@pytest.fixture(params=[DistanceWeights, DensityTaperWeights], ids=lambda c: c.__name__)
def weights_class(request) -> type[StationWeights]:
    """Each station weights model; tests build a new instance from it."""
    return request.param
