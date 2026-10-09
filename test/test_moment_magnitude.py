from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qseek.magnitudes.moment_magnitude import MomentMagnitude
from qseek.models.detection import EventDetection


@pytest.mark.asyncio
async def test_get_magnitude_without_stores() -> None:
    """MomentMagnitude implements get_magnitude of the magnitude calculators."""
    calculator = MomentMagnitude.model_construct(models=[])
    calculator._stores = []
    detection = EventDetection(
        lat=40.8,
        lon=14.1,
        time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        semblance=0.5,
        distance_border=1000.0,
    )
    with pytest.raises(ValueError, match="Could not calculate moment magnitude"):
        await calculator.get_magnitude(None, None, detection)
