from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

from pydantic import ConstrainedStr
from pyrocko.util import UnavailableDecimation

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    PhaseDescription = str

else:

    class PhaseDescription(ConstrainedStr):
        regex = r"[a-zA-Z]*:[a-zA-Z]*"


def to_path(datetime: datetime) -> str:
    return datetime.isoformat(sep="T", timespec="minutes").replace(":", "")


def to_datetime(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
    deltat = 1.0 / sampling_rate
    try:
        trace.downsample_to(deltat, demean=False, snap=True, allow_upsample_max=4)

    except UnavailableDecimation:
        logging.warn("using resample instead of decimation")
        trace.resample(deltat)


T = TypeVar("T")
P = ParamSpec("P")


def log_call(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start)
        logging.info("executed %s in %s", func.__qualname__, duration)
        return ret

    return wrapper
