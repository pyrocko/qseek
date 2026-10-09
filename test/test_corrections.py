from __future__ import annotations

import numpy as np
import pytest

from qseek.corrections.simple import SimpleCorrections
from qseek.utils import NSL


@pytest.mark.asyncio
async def test_simple_corrections() -> None:
    """The search passes the source node and nodes as keywords."""
    corrections = SimpleCorrections(stations={"XX.STA1.": {"cake:P": 0.1}})
    sta1 = NSL("XX", "STA1")
    sta2 = NSL("XX", "STA2")

    assert corrections.get_delay(sta1, "cake:P", node=None) == 0.1
    assert corrections.get_delay(sta1, "cake:S", node=None) == 0.0
    assert corrections.get_delay(sta2, "cake:P", node=None) == 0.0

    delays = await corrections.get_delays([sta1, sta2], nodes=[], phase="cake:P")
    np.testing.assert_array_equal(delays, [[0.1, 0.0]])
