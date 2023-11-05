from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Annotated, Any, AsyncIterator, Iterator, Tuple, Union

from pydantic import Field, PrivateAttr, RootModel

from lassie.images.base import ImageFunction, PickedArrival
from lassie.images.phase_net import PhaseNet, PhaseNetPick
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from datetime import timedelta

    from pyrocko.trace import Trace

    from lassie.images.base import WaveformImage
    from lassie.models.station import Stations
    from lassie.waveforms.base import WaveformBatch


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


class ImageFunctions(RootModel):
    root: list[ImageFunctionType] = [PhaseNet()]

    _queue: asyncio.Queue[Tuple[WaveformImages, WaveformBatch] | None] = PrivateAttr()
    _processed_images: int = PrivateAttr(0)

    def model_post_init(self, __context: Any) -> None:
        # Check if phases are provided twice
        phases = self.get_phases()
        if len(set(phases)) != len(phases):
            raise ValueError("A phase was provided twice")
        self._queue = asyncio.Queue(maxsize=4)

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

        async def worker() -> None:
            logger.info(
                "start pre-processing images, queue size %d", self._queue.maxsize
            )
            async for batch in batch_iterator:
                images = await self.process_traces(batch.traces)
                if self._queue.empty() and self._processed_images:
                    logger.warning("image queue ran empty, prefetching is too slow")
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
        return len(self.root)

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
        """Set the images stations."""
        for image in self:
            image.set_stations(stations)

    def cumulative_weight(self) -> float:
        return sum(image.weight for image in self)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.root
