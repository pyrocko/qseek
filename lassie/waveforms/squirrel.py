from __future__ import annotations

import asyncio
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Literal

from pydantic import (
    AwareDatetime,
    DirectoryPath,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from pyrocko.squirrel import Squirrel
from typing_extensions import Self

from lassie.models.station import Stations
from lassie.stats import Stats
from lassie.utils import datetime_now, human_readable_bytes, to_datetime
from lassie.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch
    from rich.table import Table

logger = logging.getLogger(__name__)


class SquirrelPrefetcher:
    queue: asyncio.Queue[Batch | None]
    highpass: float | None
    lowpass: float | None
    downsample_to: float | None
    load_time: timedelta = timedelta(seconds=0.0)

    _fetched_batches: int
    _task: asyncio.Task[None]

    def __init__(
        self,
        iterator: Iterator[Batch],
        queue_size: int = 8,
        downsample_to: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
    ) -> None:
        self.iterator = iterator
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.downsample_to = downsample_to
        self.highpass = highpass
        self.lowpass = lowpass
        self._fetched_batches = 0

        self._task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self) -> None:
        logger.info(
            "start prefetching data, queue size %d",
            self.queue.maxsize,
        )
        done = asyncio.Event()

        def post_processing(batch: Batch) -> Batch:
            # Filter traces in-place
            start = None

            # SeisBench would call obspy's downsampling.
            # Downsampling is much faster in Pyrocko, so we do it here.
            if self.downsample_to:
                try:
                    start = datetime_now()
                    desired_deltat = 1.0 / self.downsample_to
                    for tr in batch.traces:
                        if tr.deltat < desired_deltat:
                            tr.downsample_to(desired_deltat, allow_upsample_max=2)
                except Exception as exc:
                    logger.exception(exc)

            if self.highpass:
                start = start or datetime_now()
                for tr in batch.traces:
                    tr.highpass(4, corner=self.highpass)

            if self.lowpass:
                start = start or datetime_now()
                for tr in batch.traces:
                    tr.lowpass(4, corner=self.lowpass)

            if start:
                logger.debug("filtered waveform batch in %s", datetime_now() - start)
            return batch

        async def load_next() -> None:
            start = datetime_now()
            batch = await asyncio.to_thread(next, self.iterator, None)
            if batch is None:
                done.set()
                return
            logger.debug("read waveform batch in %s", datetime_now() - start)

            await asyncio.to_thread(post_processing, batch)
            self._fetched_batches += 1
            self.load_time = datetime_now() - start
            await self.queue.put(batch)

        while not done.is_set():
            await load_next()
        await self.queue.put(None)


class SquirrelStats(Stats):
    empty_batches: PositiveInt = 0
    short_batches: PositiveInt = 0
    time_per_batch: timedelta = timedelta(seconds=0.0)
    bytes_per_seconds: float = 0.0

    _queue: asyncio.Queue[Batch | None] | None = PrivateAttr(None)

    def set_queue(self, queue: asyncio.Queue[Batch | None]) -> None:
        self._queue = queue

    @computed_field
    @property
    def queue_size(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @computed_field
    @property
    def queue_size_max(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.maxsize

    def _populate_table(self, table: Table) -> None:
        prefix = "[bold red]" if self.queue_size <= 1 else ""
        table.add_row("Queue", f"{prefix}{self.queue_size} / {self.queue_size_max}")
        table.add_row(
            "Waveform loading",
            f"{human_readable_bytes(self.bytes_per_seconds)}/s",
        )


class PyrockoSquirrel(WaveformProvider):
    """Waveform provider using Pyrocko's Squirrel."""

    provider: Literal["PyrockoSquirrel"] = "PyrockoSquirrel"

    environment: DirectoryPath = Field(
        default=Path("."),
        description="Path to a Squirrel environment.",
    )
    waveform_dirs: list[Path] = Field(
        default=[],
        description="List of directories holding the waveform files.",
    )
    start_time: AwareDatetime | None = Field(
        default=None,
        description="Start time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601).",
    )
    end_time: AwareDatetime | None = Field(
        default=None,
        description="End time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601).",
    )

    highpass: PositiveFloat | None = Field(
        default=None,
        description="Highpass filter, corner frequency in Hz.",
    )
    lowpass: PositiveFloat | None = Field(
        default=None,
        description="Lowpass filter, corner frequency in Hz.",
    )
    downsample_to: PositiveFloat | None = Field(
        default=100.0,
        description="Downsample the data to a desired frequency",
    )

    channel_selector: str = Field(
        default="*",
        max_length=3,
        description="Channel selector for waveforms, "
        "use e.g. `EN?` for selection of all accelerometer data.",
    )
    async_prefetch_batches: PositiveInt = Field(
        default=10,
        description="Queue size for asynchronous pre-fetcher.",
    )

    _squirrel: Squirrel | None = PrivateAttr(None)
    _stations: Stations = PrivateAttr()
    _stats: SquirrelStats = PrivateAttr(default_factory=SquirrelStats)

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        if self.highpass and self.lowpass and self.highpass > self.lowpass:
            raise ValueError("freq_min must be less than freq_max")
        return self

    @field_validator("waveform_dirs")
    def check_dirs(cls, dirs: list[Path]) -> list[Path]:  # noqa: N805
        if not dirs:
            raise ValueError("no waveform directories provided!")
        return dirs

    def get_squirrel(self) -> Squirrel:
        if not self._squirrel:
            logger.debug("initializing squirrel")
            squirrel = Squirrel(str(self.environment.expanduser()))
            paths = []
            for path in self.waveform_dirs:
                if "**" in str(path):
                    paths.extend(glob.glob(str(path.expanduser()), recursive=True))
                else:
                    paths.append(str(path.expanduser()))

            squirrel.add(paths, check=False)
            self._squirrel = squirrel
        return self._squirrel

    def prepare(self, stations: Stations) -> None:
        logger.info("preparing squirrel waveform provider")
        squirrel = self.get_squirrel()
        stations.weed_from_squirrel_waveforms(squirrel)
        self._stations = stations

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_length: timedelta | None = None,
    ) -> AsyncIterator[WaveformBatch]:
        if not self._stations:
            raise ValueError("no stations provided. has prepare() been called?")

        squirrel = self.get_squirrel()
        stats = self._stats
        sq_tmin, sq_tmax = squirrel.get_time_span(["waveform"])

        start_time = start_time or self.start_time or to_datetime(sq_tmin)
        end_time = end_time or self.end_time or to_datetime(sq_tmax)

        logger.info(
            "searching time span from %s to %s (%s)",
            start_time,
            end_time,
            end_time - start_time,
        )

        iterator = squirrel.chopper_waveforms(
            tmin=start_time.timestamp(),
            tmax=end_time.timestamp(),
            tinc=window_increment.total_seconds(),
            tpad=window_padding.total_seconds(),
            want_incomplete=False,
            codes=[
                (*nsl, self.channel_selector) for nsl in self._stations.get_all_nsl()
            ],
        )
        prefetcher = SquirrelPrefetcher(
            iterator,
            queue_size=self.async_prefetch_batches,
            downsample_to=self.downsample_to,
            highpass=self.highpass,
            lowpass=self.lowpass,
        )
        stats.set_queue(prefetcher.queue)

        while True:
            start = datetime_now()
            pyrocko_batch = await prefetcher.queue.get()
            if pyrocko_batch is None:
                prefetcher.queue.task_done()
                break

            batch = WaveformBatch(
                traces=pyrocko_batch.traces,
                start_time=to_datetime(pyrocko_batch.tmin),
                end_time=to_datetime(pyrocko_batch.tmax),
                i_batch=pyrocko_batch.i,
                n_batches=pyrocko_batch.n,
            )

            batch.clean_traces()

            stats.time_per_batch = datetime_now() - start
            stats.bytes_per_seconds = (
                batch.cumulative_bytes / prefetcher.load_time.total_seconds()
            )

            if batch.is_empty():
                logger.warning("empty batch %d", batch.i_batch)
                stats.empty_batches += 1
                continue

            if min_length and batch.duration < min_length:
                logger.warning(
                    "duration of batch %d too short %s",
                    batch.i_batch,
                    batch.duration,
                )
                stats.short_batches += 1
                continue

            yield batch

            prefetcher.queue.task_done()
