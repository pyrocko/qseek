from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Awaitable, Callable, ParamSpec, TypeVar

from pydantic import BaseModel, constr
from pyrocko.util import UnavailableDecimation
from rich.logging import RichHandler

if TYPE_CHECKING:
    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)
FORMAT = "%(message)s"


PhaseDescription = Annotated[str, constr(pattern=r"[a-zA-Z]*:[a-zA-Z]*")]

CACHE_DIR = Path.home() / ".cache" / "lassie"
if not CACHE_DIR.exists():
    logger.info("creating cache dir %s", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)


class Symbols:
    Target = "🞋"
    Check = "✓"
    CheckerBoard = "🙾"


class ANSI:
    Bold = "\033[1m"
    Italic = "\033[3m"
    Underline = "\033[4m"
    Reset = "\033[0m"


def setup_rich_logging(level: int) -> None:
    logging.basicConfig(
        level=level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )


def time_to_path(datetime: datetime) -> str:
    return datetime.isoformat(sep="T", timespec="milliseconds").replace(":", "")


def to_datetime(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def downsample(trace: Trace, sampling_rate: float) -> None:
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
    """Return a human readable string representation of bytes"""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size = size / 1024.0
    return f"{size:.2f} PiB"


def datetime_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def generate_docs(model: BaseModel, exclude: dict | set | None = None) -> str:
    """Takes model and dumps markdown for documentation"""

    def generate_submodel(model: BaseModel) -> list[str]:
        lines = []
        for name, field in model.model_fields.items():
            if field.description is None:
                continue
            lines += [
                f"        - **`{name}`** *`{field.annotation}`*\n",
                f"            {field.description}",
            ]
        return lines

    model_name = model.__class__.__name__
    lines = [f"### {model_name} Module"]
    if model.__class__.__doc__ is not None:
        lines += [f"{model.__class__.__doc__}\n"]
    lines += [f'=== "Config {model_name}"']
    for name, field in model.model_fields.items():
        if field.description is None:
            continue
        lines += [
            f"    - **`{name}`**\n",
            f"        {field.description}",
        ]

    def dump_json() -> list[str]:
        dump = model.model_dump_json(by_alias=False, indent=2, exclude=exclude)
        lines = dump.split("\n")
        return [f"    {line}" for line in lines]

    lines += ['=== "JSON Block"']
    lines += [f"    ```json title='JSON block for {model_name}'"]
    lines.extend(dump_json())
    lines += ["    ```"]
    return "\n".join(lines)
