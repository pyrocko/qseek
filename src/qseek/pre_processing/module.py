from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, AsyncIterator, ClassVar, Iterator, Union

from pydantic import Field, PositiveInt, PrivateAttr, RootModel, computed_field

from qseek.pre_processing.base import BatchPreProcessing
from qseek.pre_processing.deep_denoiser import DeepDenoiser  # noqa: F401
from qseek.pre_processing.downsample import Downsample  # noqa: F401
from qseek.pre_processing.frequency_filters import (  # noqa: F401
    Bandpass,
    Highpass,
    Lowpass,
)
from qseek.stats import Stats
from qseek.utils import QUEUE_SIZE, datetime_now, human_readable_bytes

if TYPE_CHECKING:
    from rich.table import Table

    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)
BatchPreProcessingType = Annotated[
    Union[(BatchPreProcessing, *BatchPreProcessing.get_subclasses())],
    Field(..., discriminator="process"),
]


class PreProcessingStats(Stats):
    time_per_batch: timedelta = timedelta()
    bytes_per_second: float = 0.0
    _queue: asyncio.Queue[WaveformBatch | None] | None = PrivateAttr(None)

    _position: int = PrivateAttr(30)

    def set_queue(self, queue: asyncio.Queue[WaveformBatch | None] | None) -> None:
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
        if not self._queue:
            return
        prefix = "[bold red]" if self.queue_size <= 2 else ""
        table.add_row("Queue", f"{prefix}{self.queue_size} / {self.queue_size_max}")
        table.add_row(
            "Waveform processing",
            f"{human_readable_bytes(self.bytes_per_second)}/s",
        )


class PreProcessing(RootModel):
    root: list[BatchPreProcessingType] = Field(
        default=[],
        title="Pre-processing modules, evaluated in order. "
        "The first module is the first to be applied.",
    )

    _queue: asyncio.Queue[WaveformBatch | None] = PrivateAttr(
        asyncio.Queue(maxsize=QUEUE_SIZE)
    )
    _stats: ClassVar[PreProcessingStats] = PreProcessingStats()

    def __iter__(self) -> Iterator[BatchPreProcessing]:
        return iter(self.root)

    async def prepare(self) -> None:
        logger.info("preparing pre-processing modules")
        for process in self.root:
            await process.prepare()

    async def iter_batches(
        self,
        batch_iterator: AsyncIterator[WaveformBatch],
    ) -> AsyncIterator[WaveformBatch]:
        stats = self._stats
        stats.set_queue(self._queue)

        if not self.root:
            logger.debug("no pre-processing defined")
            stats.set_queue(None)
            async for batch in batch_iterator:
                yield batch
            return

        async def worker() -> None:
            async for batch in batch_iterator:
                start_time = datetime_now()
                for process in self:
                    batch = await process.process_batch(batch)
                    await asyncio.sleep(0.0)
                stats.time_per_batch = datetime_now() - start_time
                stats.bytes_per_second = (
                    batch.cumulative_bytes / stats.time_per_batch.total_seconds()
                )
                await self._queue.put(batch)

            await self._queue.put(None)

        logger.info("start pre-processing images")
        task = asyncio.create_task(worker())

        while True:
            batch = await self._queue.get()
            if batch is None:
                logger.debug("pre-processing finished")
                break
            yield batch

        logger.debug("waiting for pre-processing to finish")
        await task
