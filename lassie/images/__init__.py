from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.images.base import ImageFunction, PickedArrival
from lassie.images.phase_net import PhaseNet, PhaseNetPick

if TYPE_CHECKING:
    from datetime import timedelta

    from pyrocko.trace import Trace

    from lassie.images.base import WaveformImage
    from lassie.models.station import Stations


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[PhaseNet, ImageFunction],
    Field(discriminator="image"),
]

# Make this a Union when more picks are implemented
ImageFunctionPick = Annotated[
    Union[PhaseNetPick, PickedArrival],
    Field(discriminator="provider"),
]


class ImageFunctions(BaseModel):
    __root__: list[ImageFunctionType] = []

    async def process_traces(self, traces: list[Trace]) -> WaveformImages:
        images = []
        for function in self:
            logger.debug("calculating images from %s", function.name)
            images.extend(await function.process_traces(traces))

        return WaveformImages(__root__=images)

    def get_blinding(self) -> timedelta:
        return max(image.blinding for image in self)

    def __iter__(self) -> Iterator[ImageFunction]:
        return iter(self.__root__)


@dataclass
class WaveformImages:
    __root__: list[WaveformImage] = Field([], alias="images")

    @property
    def n_images(self) -> int:
        return len(self.__root__)

    def downsample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Downsample traces in-place.

        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False
        """
        for image in self:
            image.downsample(sampling_rate, max_normalize)

    def set_stations(self, stations: Stations) -> None:
        """Set the images stations."""
        for image in self:
            image.set_stations(stations)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.__root__
