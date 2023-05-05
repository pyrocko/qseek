from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import ConstrainedStr
from pyrocko.util import UnavailableDecimation

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    PhasePattern = str

else:

    class PhasePattern(ConstrainedStr):
        regex = r"[a-zA-Z]*:[a-zA-Z]*"


def to_datetime(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
    deltat = 1.0 / sampling_rate
    try:
        trace.downsample_to(deltat, demean=False, snap=True, allow_upsample_max=4)

    except UnavailableDecimation:
        logging.warn("using resample instead of decimation")
        trace.resample(deltat)
