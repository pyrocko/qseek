from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from matplotlib.patches import Rectangle
from pydantic import ConstrainedStr
from pyrocko.util import UnavailableDecimation

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    PhaseDescription = str

else:

    class PhaseDescription(ConstrainedStr):
        regex = r"[a-zA-Z]*:[a-zA-Z]*"


class NodeTile(Rectangle):
    def __init__(self, *args, semblance: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.semblance = semblance


def to_datetime(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
    deltat = 1.0 / sampling_rate
    try:
        trace.downsample_to(deltat, demean=False, snap=True, allow_upsample_max=4)

    except UnavailableDecimation:
        logging.warn("using resample instead of decimation")
        trace.resample(deltat)
