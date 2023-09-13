from __future__ import annotations

import asyncio
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Literal

from pydantic import AwareDatetime, PositiveInt, PrivateAttr, constr, model_validator
from pyrocko.squirrel import Squirrel
from typing_extensions import Self

from lassie.models.station import Stations
from lassie.utils import datetime_now, to_datetime
from lassie.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch

logger = logging.getLogger(__name__)


class SquirrelPrefetcher:
    def __init__(self, iterator: Iterator[Batch], queue_size: int = 4) -> None:
        self.iterator = iterator
        self.queue: asyncio.Queue[Batch | None] = asyncio.Queue(maxsize=queue_size)

        self._task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self) -> None:
        logger.info("start prefetching squirrel data")
        while True:
            start = datetime_now()
            batch = await asyncio.to_thread(lambda: next(self.iterator, None))
            logger.debug("prefetched waveforms in %s", datetime_now() - start)
            if batch is None:
                logger.debug("squirrel prefetcher finished")
                await self.queue.put(None)
                break
            await self.queue.put(batch)


class PyrockoSquirrel(WaveformProvider):
    provider: Literal["PyrockoSquirrel"] = "PyrockoSquirrel"

    environment: Path = Path(".")
    waveform_dirs: list[Path] = []
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None

    channel_selector: constr(max_length=3) = "*"
    async_prefetch_batches: PositiveInt = 4

    _squirrel: Squirrel | None = PrivateAttr(None)
    _stations: Stations = PrivateAttr()

    @model_validator(mode="after")
    def _validate_time_span(self) -> Self:  # noqa: N805
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

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
    ) -> AsyncIterator[WaveformBatch]:
        if not self._stations:
            raise ValueError("no stations provided. has prepare() been called?")

        squirrel = self.get_squirrel()
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
        prefetcher = SquirrelPrefetcher(iterator, self.async_prefetch_batches)

        while True:
            batch = await prefetcher.queue.get()
            if batch is None:
                prefetcher.queue.task_done()
                break

            yield WaveformBatch(
                traces=batch.traces,
                start_time=to_datetime(batch.tmin),
                end_time=to_datetime(batch.tmax),
                i_batch=batch.i,
                n_batches=batch.n,
            )

            prefetcher.queue.task_done()
