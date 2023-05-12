from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.images.base import ImageFunction
from lassie.images.phase_net import PhaseNet

if TYPE_CHECKING:
    from datetime import timedelta

    from pyrocko.trace import Trace

    from lassie.images.base import WaveformImage


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[PhaseNet, ImageFunction],
    Field(discriminator="image"),
]


class ImageFunctions(BaseModel):
    __root__: list[ImageFunctionType] = []

    def process_traces(self, traces: list[Trace]) -> WaveformImages:
        images = []
        for function in self:
            logger.debug("calculating images from %s", function.name)
            images.extend(function.process_traces(traces))

        return WaveformImages(__root__=images)

    def get_blinding(self) -> timedelta:
        return max(image.blinding for image in self)

    def __iter__(self) -> Iterator[ImageFunction]:
        yield from self.__root__


@dataclass
class WaveformImages:
    __root__: list[WaveformImage] = Field([], alias="images")

    @property
    def n_images(self) -> int:
        return len(self.__root__)

    def get_max_samples(self) -> int:
        """Get maximum number of samples of all images.

        Returns:
            int: Number of samples.
        """
        return max(image.get_max_samples() for image in self)

    def downsample(self, sampling_rate: float) -> None:
        """Downsample traces in-place.
        Args:
            sampling_rate (float): Sampling rate in Hz.
        """
        for image in self:
            image.downsample(sampling_rate)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.__root__
