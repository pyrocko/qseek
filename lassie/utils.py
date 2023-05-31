from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, ParamSpec, TypeVar

from pydantic import ConstrainedStr
from pyrocko.util import UnavailableDecimation

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    PhaseDescription = str

else:

    class PhaseDescription(ConstrainedStr):
        regex = r"[a-zA-Z]*:[a-zA-Z]*"


logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache"
if not CACHE_DIR.exists():
    logger.info("creating cache dir %s", CACHE_DIR)
    CACHE_DIR.mkdir()


def to_path(datetime: datetime) -> str:
    return datetime.isoformat(sep="T", timespec="minutes").replace(":", "")


def to_datetime(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
    deltat = 1.0 / sampling_rate
    try:
        trace.downsample_to(deltat, demean=False, snap=True, allow_upsample_max=4)

    except UnavailableDecimation:
        logger.warn("using resample instead of decimation")
        trace.resample(deltat)


T = TypeVar("T")
P = ParamSpec("P")


def log_call(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start)
        logger.info("executed %s in %s", func.__qualname__, duration)
        return ret

    return wrapper


def alog_call(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = await func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start)
        logger.info("executed %s in %s", func.__qualname__, duration)
        return ret

    return wrapper
