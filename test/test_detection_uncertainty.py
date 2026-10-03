from __future__ import annotations

import numpy as np
import pytest

from qseek.models.detection_uncertainty import DetectionUncertainty


def test_uncertainty_is_the_extent() -> None:
    uncertainty = DetectionUncertainty(
        east=(-300.0, 100.0),
        north=(-200.0, 100.0),
        depth=(-150.0, 250.0),
    )
    assert uncertainty.horizontal == pytest.approx(np.hypot(400.0, 300.0))
    assert uncertainty.vertical == pytest.approx(400.0)
    assert uncertainty.total == pytest.approx(np.sqrt(400.0**2 + 300.0**2 + 400.0**2))


def test_symmetric_uncertainty() -> None:
    """A symmetric extent around the source node is not zero."""
    uncertainty = DetectionUncertainty(
        east=(-200.0, 200.0),
        north=(-200.0, 200.0),
        depth=(-200.0, 200.0),
    )
    assert uncertainty.horizontal == pytest.approx(np.hypot(400.0, 400.0))
    assert uncertainty.total == pytest.approx(np.sqrt(3 * 400.0**2))


def test_serialized_uncertainty() -> None:
    uncertainty = DetectionUncertainty(
        east=(-300.0, 100.0),
        north=(-200.0, 100.0),
        depth=(-150.0, 250.0),
    )
    dump = uncertainty.model_dump()
    assert dump["horizontal"] == pytest.approx(500.0)
    assert dump["vertical"] == pytest.approx(400.0)
