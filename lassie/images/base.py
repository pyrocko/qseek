from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from lassie.utils import downsample, to_datetime

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.trace import Trace


class ImageFunction(BaseModel):
    def process_traces(self, traces: list[Trace]) -> WaveformImages:
        ...


@dataclass
class WaveformImages:
    name: str
    images: list[WaveformImage]

    def downsample(self, sampling_rate: float) -> None:
        """Downsample in-place.

        Args:
            sampling_rate (float): Sampling rate in Hz.
        """
        for image in self.images:
            image.downsample(sampling_rate)


@dataclass
class WaveformImage:
    phase: str
    traces: list[Trace]

    @property
    def start_time(self) -> datetime:
        return to_datetime(min((tr.tmin for tr in self.traces)))

    @property
    def end_time(self) -> datetime:
        return to_datetime(max((tr.tmax for tr in self.traces)))

    def downsample(self, sampling_rate: float) -> None:
        """Downsample in-place.

        Args:
            sampling_rate (float): Sampling rate in Hz.
        """
        for tr in self.traces:
            downsample(tr, sampling_rate)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        snuffle(self.traces)
