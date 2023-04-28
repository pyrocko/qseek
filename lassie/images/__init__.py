from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.images.base import ImageFunction
from lassie.images.phase_net import PhaseNet

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from lassie.images.base import WaveformImage


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[PhaseNet, ImageFunction],
    Field(discriminator="image"),
]


class ImageFunctions(BaseModel):
    __root__: list[ImageFunctionType] = Field([], alias="functions")

    def process_traces(self, traces: list[Trace]) -> WaveformImages:
        images = []
        for function in self:
            logger.debug("calculating images from %s", function.name)
            images.extend(function.process_traces(traces))

        return WaveformImages(images=images)

    def __iter__(self) -> Iterator[ImageFunction]:
        yield from self.__root__


@dataclass
class WaveformImages:
    images: list[WaveformImage]

    def downsample(self, sampling_rate: float) -> None:
        """Downsample in-place.
        Args:
            sampling_rate (float): Sampling rate in Hz.
        """
        for image in self:
            image.downsample(sampling_rate)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.images
