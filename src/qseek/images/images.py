from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from typing import TYPE_CHECKING, Annotated, Any, AsyncIterator, Iterator, Tuple, Union

from pydantic import Field, PositiveInt, PrivateAttr, RootModel, computed_field

from qseek.images.base import ImageFunction, PickedArrival
from qseek.images.phase_net import PhaseNet, PhaseNetPick
from qseek.stats import Stats
from qseek.utils import PhaseDescription, datetime_now, human_readable_bytes

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table

    from qseek.images.base import WaveformImage
    from qseek.models.station import Stations
    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[PhaseNet, ImageFunction],
    Field(..., discriminator="image"),
]

# Make this a Union when more picks are implemented
ImageFunctionPick = Annotated[
    Union[PhaseNetPick, PickedArrival],
    Field(..., discriminator="provider"),
]


class ImageFunctionsStats(Stats):
    time_per_batch: timedelta = timedelta()
    bytes_per_second: float = 0.0

    _queue: asyncio.Queue[WaveformImages | None] | None = PrivateAttr(None)

    def set_queue(self, queue: asyncio.Queue[WaveformImages | None]) -> None:
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
        prefix = "[bold red]" if self.queue_size <= 2 else ""
        table.add_row("Queue", f"{prefix}{self.queue_size} / {self.queue_size_max}")
        table.add_row(
            "Waveform processing",
            f"{human_readable_bytes(self.bytes_per_second)}/s",
        )


class ImageFunctions(RootModel):
    root: list[ImageFunctionType] = [PhaseNet()]

    _queue: asyncio.Queue[Tuple[WaveformImages, WaveformBatch] | None] = PrivateAttr()
    _processed_images: int = PrivateAttr(0)
    _stats = PrivateAttr(default_factory=ImageFunctionsStats)

    def model_post_init(self, __context: Any) -> None:
        # Check if phases are provided twice
        phases = self.get_phases()
        if len(set(phases)) != len(phases):
            raise ValueError("A phase was provided twice")
        self._queue = asyncio.Queue(maxsize=10)
        self._stats.set_queue(self._queue)

    async def process_traces(self, traces: list[Trace]) -> WaveformImages:
        images = []
        for function in self:
            logger.debug("calculating images from %s", function.name)
            images.extend(await function.process_traces(traces))

        return WaveformImages(root=images)

    async def iter_images(
        self,
        batch_iterator: AsyncIterator[WaveformBatch],
    ) -> AsyncIterator[Tuple[WaveformImages, WaveformBatch]]:
        """Iterate over images from batches.

        Args:
            batches (AsyncIterator[Batch]): Async iterator over batches.

        Yields:
            AsyncIterator[WaveformImages]: Async iterator over images.
        """

        stats = self._stats

        async def worker() -> None:
            logger.info(
                "start pre-processing images, queue size %d", self._queue.maxsize
            )
            async for batch in batch_iterator:
                start_time = datetime_now()
                images = await self.process_traces(batch.traces)
                stats.time_per_batch = datetime_now() - start_time
                stats.bytes_per_second = (
                    batch.cumulative_bytes / stats.time_per_batch.total_seconds()
                )
                self._processed_images += 1
                await self._queue.put((images, batch))

            await self._queue.put(None)

        task = asyncio.create_task(worker())

        while True:
            ret = await self._queue.get()
            if ret is None:
                logger.debug("image function finished")
                break
            yield ret

        await task

    def get_phases(self) -> tuple[PhaseDescription, ...]:
        """Get all phases that are available in the image functions.

        Returns:
            tuple[str, ...]: All available phases.
        """
        return tuple(chain.from_iterable(image.get_provided_phases() for image in self))

    def get_blinding(self) -> timedelta:
        return max(image.blinding for image in self)

    def __iter__(self) -> Iterator[ImageFunction]:
        return iter(self.root)


@dataclass
class WaveformImages:
    root: list[WaveformImage] = Field([], alias="images")

    @property
    def n_images(self) -> int:
        """Number of image functions."""
        return len(self.root)

    @property
    def n_stations(self) -> int:
        """Number of stations in the images."""
        return max(0, *(image.stations.n_stations for image in self if image.stations))

    def downsample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Downsample traces in-place.

        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False
        """
        for image in self:
            image.downsample(sampling_rate, max_normalize)

    def apply_exponent(self, exponent: float) -> None:
        """Apply exponent to all images.

        Args:
            exponent (float): Exponent to apply.
        """
        for image in self:
            image.apply_exponent(exponent)

    def set_stations(self, stations: Stations) -> None:
        """Set the images stations.

        Args:
            stations (Stations): Stations to set.
        """
        for image in self:
            image.set_stations(stations)

    def cumulative_weight(self) -> float:
        """Get the cumulative weight of all images."""
        return sum(image.weight for image in self)

    def snuffle(self) -> None:
        """Open Pyrocko Snuffler on the image traces."""
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.root