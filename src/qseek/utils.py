from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
)

import numpy as np
from pydantic import AfterValidator, ByteSize, constr
from pyrocko.util import UnavailableDecimation
from rich.logging import RichHandler

if TYPE_CHECKING:
    from contextvars import Context

    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)
FORMAT = "%(message)s"


PhaseDescription = Annotated[str, constr(pattern=r"[a-zA-Z]*:[a-zA-Z]*")]

CACHE_DIR = Path.home() / ".cache" / "qseek"
if not CACHE_DIR.exists():
    logger.info("creating cache dir %s", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)


class Symbols:
    Target = "🞋"
    Check = "✓"
    CheckerBoard = "🙾"
    Cross = "✗"


def setup_rich_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )


class BackgroundTasks:
    tasks: ClassVar[set[asyncio.Task]] = set()

    @classmethod
    def create_task(
        cls,
        coro: Coroutine,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name, context=context)
        cls.tasks.add(task)
        task.add_done_callback(cls.tasks.remove)
        return task

    @classmethod
    def cancel_all(cls) -> None:
        for task in cls.tasks:
            task.cancel()

    @classmethod
    async def wait_all(cls) -> None:
        await asyncio.gather(*cls.tasks)


class NSL(NamedTuple):
    network: str
    station: str
    location: str

    @property
    def pretty(self) -> str:
        return ".".join(self)

    def match(self, other: NSL) -> bool:
        """
        Check if the current NSL object matches another NSL object.

        Args:
            other (NSL): The NSL object to compare with.

        Returns:
            bool: True if the objects match, False otherwise.
        """
        if other.location:
            return self == other
        if other.station:
            return self.network == other.network and self.station == other.station
        return self.network == other.network

    @classmethod
    def parse(cls, nsl: str) -> NSL:
        """
        Parse the given NSL string and return an NSL object.

        Args:
            nsl (str): The NSL string to parse.

        Returns:
            NSL: The parsed NSL object.

        Raises:
            ValueError: If the NSL string is empty or invalid.
        """
        if not nsl:
            raise ValueError("invalid empty NSL")
        try:
            net, sta, loc, *_ = nsl.split(".")
        except ValueError:
            try:
                net, sta = nsl.split(".")
                loc = ""
            except ValueError:
                net = nsl
                sta = ""
                loc = ""
        return cls(network=net, station=sta, location=loc)


class _Range(NamedTuple):
    min: float
    max: float

    def inside(self, value: float) -> bool:
        """
        Check if a value is inside the range.

        Args:
            value (float): The value to check.

        Returns:
            bool: True if the value is inside the range, False otherwise.
        """
        return self.min <= value <= self.max

    @classmethod
    def from_list(cls, array: np.ndarray | list[float]) -> _Range:
        """
        Create a Range object from a numpy array.

        Parameters:
        - array: numpy.ndarray
            The array from which to create the Range object.

        Returns:
        - _Range: The created Range object.
        """
        return cls(min=np.min(array), max=np.max(array))


def _range_validator(v: _Range) -> _Range:
    if v.min > v.max:
        raise ValueError(f"Bad range {v}, must be (min, max)")
    return v


Range = Annotated[_Range, AfterValidator(_range_validator)]


def time_to_path(datetime: datetime) -> str:
    """
    Converts a datetime object to a string representation of a file path.

    Args:
        datetime (datetime): The datetime object to convert.

    Returns:
        str: The string representation of the file path.
    """
    return datetime.isoformat(sep="T", timespec="milliseconds").replace(":", "")


def to_datetime(time: float) -> datetime:
    """
    Convert a UNIX timestamp to a datetime object in UTC timezone.

    Args:
        time (float): The UNIX timestamp to convert.

    Returns:
        datetime: The corresponding datetime object in UTC timezone.
    """
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
    """
    Downsamples the given trace to the specified sampling rate in-place.

    Args:
        trace (Trace): The trace to be downsampled.
        sampling_rate (float): The desired sampling rate.
    """
    deltat = 1.0 / sampling_rate

    if trace.deltat == deltat:
        return

    try:
        trace.downsample_to(deltat, demean=False, snap=False, allow_upsample_max=4)
    except UnavailableDecimation:
        logger.warning("using resample instead of decimation")
        trace.resample(deltat)


T = TypeVar("T")
P = ParamSpec("P")


def log_call(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start)
        logger.debug("executed %s in %s", func.__qualname__, duration)
        return ret

    return wrapper


def alog_call(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = await func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start)
        logger.debug("executed %s in %s", func.__qualname__, duration)
        return ret

    return wrapper


def human_readable_bytes(size: int | float) -> str:
    """
    Convert a size in bytes to a human-readable string representation.

    Args:
        size (int | float): The size in bytes.

    Returns:
        str: The human-readable string representation of the size.

    """
    return ByteSize(size).human_readable()


def datetime_now() -> datetime:
    """
    Get the current datetime in UTC timezone.

    Returns:
        datetime: The current datetime in UTC timezone.
    """
    return datetime.now(tz=timezone.utc)


def get_cpu_count() -> int:
    """
    Get the number of CPUs available for the current job/task.

    The function first checks if the environment variable SLURM_CPUS_PER_TASK is set.
    If it is set, the value is returned as the number of CPUs.

    If SLURM_CPUS_PER_TASK is not set, the function then checks if the environment
    variable PBS_NUM_PPN is set. If it is set, the value is returned as the number of
    CPUs.

    If neither SLURM_CPUS_PER_TASK nor PBS_NUM_PPN is set, the function returns the
    number of CPUs available on the system using os.cpu_count().

    Returns:
        int: The number of CPUs available for the current job/task.
    """
    return int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            os.environ.get(
                "PBS_NUM_PPN",
                os.cpu_count() or 0,
            ),
        )
    )


def filter_clipped_traces(
    traces: list[Trace],
    counts_threshold: int = 20,
    max_bits: tuple[int, ...] = (24, 32),
) -> list[Trace]:
    """
    Filters out clipped traces from the given list of traces.

    Args:
        traces (list[Trace]): The list of traces to filter.
        counts_threshold (int, optional): The threshold for the distance between the
            maximum value of a trace and the clip value. Defaults to 20.
        max_bits (tuple[int, ...], optional): The clip bits to check for.
            Defaults to (24, 32).

    Raises:
        TypeError: If a trace is not of type int, np.int32 or np.int64.

    Returns:
        list[Trace]: The filtered list of traces.
    """
    for tr in traces.copy():
        if tr.ydata is None:
            continue
        if tr.ydata.dtype not in (int, np.int32, np.int64):
            raise TypeError(f"trace {tr.nslc_id} has invalid dtype {tr.ydata.dtype}")

        max_val = np.abs(tr.ydata).max()
        for bits in max_bits:
            clip_value = 2 ** (bits - 1) - 1
            distance_counts = abs(max_val - clip_value)
            if distance_counts < counts_threshold:
                logger.warning(
                    "trace %s likely clipped, distance to %d bits clip are %d counts",
                    ".".join(tr.nslc_id),
                    bits,
                    distance_counts,
                )
                traces.remove(tr)
    return traces


def camel_case_to_snake_case(name: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        name (str): The camel case string to be converted.

    Returns:
        str: The snake case string.

    Example:
        >>> camel_case_to_snake_case("camelCaseString")
        'camel_case_string'
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_insights() -> None:
    """
    Imports the qseek.insights package if available.

    This function attempts to import the qseek.insights package and logs a debug message
    indicating whether the package was successfully imported or not.

    Raises:
        ImportError: If the qseek.insights package is not installed.
    """
    try:
        import qseek.insights  # noqa: F401

        logger.info("loaded qseek.insights package")
    except ImportError as exc:
        logger.warning("package qseek.insights not installed", exc_info=exc)


MeasurementUnit = Literal[
    "displacement",
    "velocity",
    "acceleration",
]


@dataclass
class ChannelSelector:
    channels: str
    number_channels: int
    normalize: bool = False

    def get_traces(self, traces: list[Trace]) -> list[Trace]:
        """
        Filter and normalize a list of traces based on the specified channels.

        Args:
            traces (list[Trace]): The list of traces to filter.

        Returns:
            list[Trace]: The filtered and normalized list of traces.

        Raises:
            KeyError: If the number of channels in the filtered list does not match
                the expected number of channels.
        """
        traces = [tr for tr in traces if tr.channel[-1] in self.channels]
        if len(traces) != self.number_channels:
            raise KeyError(
                f"cannot get {self.number_channels} channels"
                f" for selector {self.channels}"
                f" available: {', '.join('.'.join(tr.nslc_id) for tr in traces)}"
            )
        if self.normalize:
            traces_norm = traces[0].copy()
            traces_norm.ydata = np.linalg.norm(
                np.array([tr.ydata for tr in traces]), axis=0
            )
            return [traces_norm]
        return traces

    __call__ = get_traces


class ChannelSelectors:
    All = ChannelSelector("ENZ0123RT", 3)
    HorizontalAbs = ChannelSelector("EN123RT", 2, normalize=True)
    Horizontal = ChannelSelector("EN123RT", 2)
    Vertical = ChannelSelector("Z0", 1)
    NorthEast = ChannelSelector("NE", 2)
