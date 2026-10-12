from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, AsyncIterator, ClassVar, Iterator, Union

from pydantic import Field, RootModel

from qseek.pre_processing.base import BatchPreProcessing
from qseek.pre_processing.deep_denoiser import DeepDenoiser
from qseek.pre_processing.frequency_filters import (
    Bandpass,
    Highpass,
    Lowpass,
)
from qseek.pre_processing.resample import Downsample, Resample
from qseek.stats import QueueStats
from qseek.utils import (
    QUEUE_SIZE,
    aclosing,
    datetime_now,
    prefetch,
)

if TYPE_CHECKING:
    from rich.table import Table

    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)
# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type BatchPreProcessingType = BatchPreProcessing
else:
    type BatchPreProcessingType = Annotated[
        Union[BatchPreProcessing.get_subclasses()],
        Field(discriminator="process"),
    ]


class PreProcessingStats(QueueStats):
    time_per_batch: timedelta = timedelta()
    bytes_per_second: float = 0.0

    _position = 30
    _show_header = False

    def _populate_table(self, table: Table) -> None:
        if self._queue is None:
            return
        self._add_queue_row(table, "Waveform PreProcessing", self.bytes_per_second)


class PreProcessing(RootModel):
    root: list[BatchPreProcessingType] = Field(
        default=[],
        title="Pre-processing modules, evaluated in order. "
        "The first module is the first to be applied.",
    )

    _stats: ClassVar[PreProcessingStats] = PreProcessingStats()

    # BaseModel.__iter__ yields the fields, this yields the items
    def __iter__(self) -> Iterator[BatchPreProcessing]:  # ty: ignore[invalid-method-override]
        return iter(self.root)

    async def prepare(self) -> None:
        logger.info("preparing pre-processing modules")
        for process in self.root:
            await process.prepare()

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        for process in self:
            batch = await process.process_batch(batch)
        return batch

    def iter_batches(
        self,
        batch_iterator: AsyncIterator[WaveformBatch],
    ) -> AsyncIterator[WaveformBatch]:
        """Iterate over the pre-processed batches.

        The batches are processed in a background task, ahead of the consumer.
        Errors are raised to the consumer, and the background task is cancelled
        when the consumer stops iterating.

        Args:
            batch_iterator (AsyncIterator[WaveformBatch]): Async iterator over
                batches.

        Returns:
            AsyncIterator[WaveformBatch]: The pre-processed batches.
        """
        if not self.root:
            logger.debug("no pre-processing defined")
            self._stats.set_queue(None)
            return batch_iterator

        queue: asyncio.Queue[WaveformBatch] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._stats.set_queue(queue)
        logger.info("start pre-processing waveforms")
        return prefetch(self._process_batches(batch_iterator), queue, "pre-processing")

    async def _process_batches(
        self,
        batch_iterator: AsyncIterator[WaveformBatch],
    ) -> AsyncIterator[WaveformBatch]:
        stats = self._stats
        async with aclosing(batch_iterator):
            async for batch in batch_iterator:
                start_time = datetime_now()
                for process in self:
                    batch = await process.process_batch(batch)
                    await asyncio.sleep(0.0)
                stats.time_per_batch = datetime_now() - start_time
                stats.bytes_per_second = (
                    batch.nbytes / stats.time_per_batch.total_seconds()
                )
                yield batch
        logger.debug("pre-processing finished")


__all__ = [
    "Bandpass",
    "DeepDenoiser",
    "Downsample",
    "Highpass",
    "Lowpass",
    "PreProcessing",
    "Resample",
]
