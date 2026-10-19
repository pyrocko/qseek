from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import math
from datetime import datetime, timedelta
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Iterator, Literal, Self, Sequence

from pydantic import (
    Field,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pyrocko.squirrel import Squirrel

from qseek.models.station import StationInventory
from qseek.stats import QueueStats
from qseek.types import DirectoryPath
from qseek.utils import (
    NSL,
    QUEUE_SIZE,
    DateTime,
    datetime_now,
    prefetch,
    to_datetime,
)
from qseek.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch
    from pyrocko.trace import Trace
    from rich.table import Table

logger = logging.getLogger(__name__)

_SQUIRREL_SEM = asyncio.Semaphore(8)


async def _load_batches(
    iterator: Iterator[Batch],
) -> AsyncGenerator[tuple[Batch, timedelta]]:
    """Load the batches of a Squirrel chopper in a thread.

    Yields:
        tuple[Batch, timedelta]: The batch and the time it took to load it.
    """

    def next_batch() -> Batch | None:
        return next(iterator, None)

    n_batches = 0
    while True:
        start_load = datetime_now()
        logger.debug("loading waveform batch %d", n_batches)
        batch = await asyncio.to_thread(next_batch)
        if batch is None:
            logger.debug("done loading waveforms")
            return
        load_time = datetime_now() - start_load
        logger.debug("read waveform batch in %s", load_time)
        n_batches += 1
        yield batch, load_time


class SquirrelStats(QueueStats):
    empty_batches: int = 0
    short_batches: int = 0
    time_per_batch: timedelta = timedelta(seconds=0.0)
    bytes_per_seconds: float = 0.0

    _position = 20
    _show_header = False

    def _populate_table(self, table: Table) -> None:
        self._add_queue_row(table, "Waveform loading", self.bytes_per_seconds)


class PyrockoSquirrel(WaveformProvider):
    """Load unstructured waveform data through Pyrocko's Squirrel."""

    provider: Literal["PyrockoSquirrel"] = "PyrockoSquirrel"

    environment: DirectoryPath | None = Field(
        default=None,
        description="Path to a [Pyrocko Squirrel environment](https://pyrocko.org/docs/current/apps/squirrel/reference/squirrel_init.html).",
    )
    persistent: str | None = Field(
        default=None,
        description="Name of the [Squirrel's persistent collection](https://pyrocko.org/docs/current/apps/squirrel/reference/squirrel_persistent.html)"
        " for faster loading of large data sets.",
    )
    waveform_dirs: list[Path] = Field(
        default=[Path("./data")],
        description="List of directories holding the waveform files.",
    )
    start_time: DateTime | None = Field(
        default=None,
        description="Start time for the search as in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) as a "
        "date or datetime including timezone. "
        "E.g. `today`, `yesterday`, `2025-12-24` or `2024-12-30T00:00:00Z`.",
    )
    end_time: DateTime | None = Field(
        default=None,
        description=(
            "End time for the search in [ISO "
            "8601](https://en.wikipedia.org/wiki/ISO_8601), as a date or a datetime "
            "with time zone, e.g. `today`, `yesterday`, `2025-12-24` or "
            "`2024-12-30T00:00:00Z`."
        ),
    )

    n_threads: PositiveInt = Field(
        default=8,
        description="Number of threads for loading waveforms,"
        " important for large data sets.",
    )
    watch_waveforms: bool | timedelta = Field(
        default=False,
        description=(
            "Watch the waveform directories for new data. `true` checks every ten "
            "minutes, a duration sets the interval."
        ),
    )
    queue_size: int = Field(
        default=QUEUE_SIZE,
        description="Size of the internal queue for prefetching waveform batches.",
    )

    _squirrel: Squirrel | None = PrivateAttr(None)
    _stations: StationInventory | None = PrivateAttr(None)
    _stats: SquirrelStats = PrivateAttr(default_factory=SquirrelStats)

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        if not self.waveform_dirs and not self.persistent:
            raise ValueError("no waveform directories or persistent selection provided")
        if self.watch_waveforms and not self.waveform_dirs:
            raise ValueError("watch_waveforms requires waveform_dirs")
        if self.watch_waveforms and self.end_time:
            raise ValueError("watch_waveforms does not support end_time")
        return self

    @field_validator("watch_waveforms", mode="after")
    @classmethod
    def _validate_watch(cls, value: bool | timedelta) -> timedelta | bool:
        if value is True:
            return timedelta(minutes=10)
        if isinstance(value, timedelta) and value < timedelta(minutes=10):
            raise ValueError("watch_waveform_dirs must be at least 10 minutes")
        return value

    def get_squirrel(self) -> Squirrel:
        if not self._squirrel:
            logger.info(
                "loading squirrel environment from %s",
                self.environment or "home directory",
            )
            squirrel_args = {
                "env": str(self.environment.expanduser()) if self.environment else None,
                "persistent": self.persistent,
                "n_threads": self.n_threads,
                "caching": "lru",
            }

            squirrel_sig = signature(Squirrel)
            if "n_threads" not in squirrel_sig.parameters:
                logger.warning("Squirrel does not support n_threads")
                squirrel_args.pop("n_threads")
            if "caching" not in squirrel_sig.parameters:
                logger.warning("Squirrel does not support caching='lru'")
                squirrel_args.pop("caching")

            squirrel = Squirrel(**squirrel_args)

            if self.waveform_dirs:
                self.scan_waveform_dirs(squirrel)

            self._squirrel = squirrel
        return self._squirrel

    def scan_waveform_dirs(self, squirrel: Squirrel) -> None:
        if not self.waveform_dirs:
            return

        logger.info(
            "scanning waveform directories %s",
            ", ".join(map(str, self.waveform_dirs)),
        )
        paths = []
        for path in self.waveform_dirs:
            if "**" in str(path):
                paths.extend(glob.glob(str(path.expanduser()), recursive=True))
            else:
                paths.append(str(path.expanduser()))
        squirrel.add(paths, check=False)

    async def prepare(self, stations: StationInventory) -> None:
        logger.info("preparing squirrel waveform provider")
        self._stations = stations

    def available_nsls(self) -> set[NSL]:
        squirrel = self.get_squirrel()
        available_codes = squirrel.get_codes(kind="waveform")
        return {NSL(*code[0:3]) for code in available_codes}

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        min_length: timedelta | None = None,
        end_time: datetime | None = None,
    ) -> AsyncGenerator[WaveformBatch]:
        stations = self._stations
        if not stations:
            raise ValueError("no stations provided. has prepare() been called?")

        squirrel = self.get_squirrel()
        stats = self._stats

        if self.watch_waveforms:
            logger.info("scanning for new waveforms every %s", self.watch_waveforms)

        def prefetch_batches(
            chop_start_time: datetime | None = None,
            chop_end_time: datetime | None = None,
            trim_end: timedelta | None = None,
        ) -> AsyncGenerator[tuple[Batch, timedelta]]:
            sq_tmin, sq_tmax = map(to_datetime, squirrel.get_time_span(["waveform"]))
            chop_start_time = chop_start_time or self.start_time or sq_tmin
            chop_end_time = chop_end_time or self.end_time or sq_tmax
            if trim_end:
                chop_end_time -= trim_end

            logger.info(
                "searching time span from %s to %s (%s)",
                chop_start_time,
                chop_end_time,
                chop_end_time - chop_start_time,
            )

            iterator = squirrel.chopper_waveforms(
                tmin=(chop_start_time + window_padding).timestamp(),
                tmax=(chop_end_time - window_padding).timestamp(),
                tinc=window_increment.total_seconds(),
                tpad=window_padding.total_seconds(),
                want_incomplete=False,
                codes=[(*nsl, "*") for nsl in stations.get_nsls()],
                channel_priorities=self.channel_selector,
            )
            queue: asyncio.Queue[tuple[Batch, timedelta]] = asyncio.Queue(
                maxsize=self.queue_size
            )
            stats.set_queue(queue)
            logger.info("start prefetching waveforms, queue size %d", queue.maxsize)
            return prefetch(_load_batches(iterator), queue, name="Squirrel loading")

        pyrocko_batches = prefetch_batches(
            chop_start_time=start_time, chop_end_time=end_time
        )
        last_batch_end_time = None

        while True:
            start = datetime_now()
            async with contextlib.aclosing(pyrocko_batches):
                async for pyrocko_batch, load_time in pyrocko_batches:
                    batch = WaveformBatch(
                        traces=pyrocko_batch.traces,
                        start_time=to_datetime(pyrocko_batch.tmin),
                        end_time=to_datetime(pyrocko_batch.tmax),
                        i_batch=pyrocko_batch.i,
                        n_batches=pyrocko_batch.n,
                    )
                    batch.clean_traces()

                    stats.time_per_batch = datetime_now() - start
                    stats.bytes_per_seconds = batch.nbytes / load_time.total_seconds()

                    if not batch.is_healthy():
                        logger.warning(
                            "unhealthy batch %d - %s",
                            batch.i_batch,
                            batch.start_time,
                        )
                        stats.empty_batches += 1
                        start = datetime_now()
                        continue

                    if min_length and batch.duration < min_length:
                        logger.warning(
                            "duration of batch %d too short %s",
                            batch.i_batch,
                            batch.duration,
                        )
                        stats.short_batches += 1
                        start = datetime_now()
                        continue

                    last_batch_end_time = batch.end_time
                    yield batch
                    start = datetime_now()

            if not isinstance(self.watch_waveforms, timedelta):
                logger.debug("no more waveforms to load")
                break

            logger.info("re-scanning waveform directories in %s", self.watch_waveforms)
            await asyncio.sleep(self.watch_waveforms.total_seconds())
            self.scan_waveform_dirs(squirrel)
            pyrocko_batches = prefetch_batches(
                chop_start_time=last_batch_end_time,
                chop_end_time=None,
                trim_end=timedelta(seconds=30),  # Trim as SeedLink is slow!
            )

    async def get_traces(
        self,
        nsls: Sequence[NSL],
        start_time: datetime,
        end_time: datetime,
        channel_priorities: Sequence[str] | None = None,
        want_incomplete: bool = False,
    ) -> list[Trace]:
        squirrel = self.get_squirrel()

        accessor_id = "waveforms.squirrel"
        nslc_ids = [(*nsl, "*") for nsl in nsls]
        async with _SQUIRREL_SEM:
            traces = await asyncio.to_thread(
                squirrel.get_waveforms,
                codes=nslc_ids,
                tmin=start_time.timestamp(),
                tmax=end_time.timestamp(),
                snap=(math.ceil, math.floor),
                accessor_id=accessor_id,
                want_incomplete=want_incomplete,
                channel_priorities=channel_priorities or self.channel_selector,
            )
        squirrel.advance_accessor(accessor_id, cache_id="waveform")
        return traces
