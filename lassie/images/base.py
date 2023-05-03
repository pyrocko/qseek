from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

from lassie.utils import downsample, to_datetime

if TYPE_CHECKING:
    from pyrocko.trace import Trace


class ImageFunction(BaseModel):
    image: Literal["base"] = "base"

    def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_available_phases(self) -> tuple[str]:
        ...


@dataclass
class WaveformImage:
    image_function: ImageFunction
    phase: str
    traces: list[Trace]

    @property
    def start_time(self) -> datetime:
        return to_datetime(min((tr.tmin for tr in self.traces)))

    @property
    def end_time(self) -> datetime:
        return to_datetime(max((tr.tmax for tr in self.traces)))

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.delta_t

    @property
    def delta_t(self) -> float:
        return self.traces[0].deltat

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    def get_all_nsl(self) -> tuple[tuple[str, str, str]]:
        return tuple((tr.network, tr.station, tr.location) for tr in self.traces)

    def downsample(self, sampling_rate: float) -> None:
        """Downsample in-place.
        Args:
            sampling_rate (float): Sampling rate in Hz.
        """
        for tr in self.traces:
            downsample(tr, sampling_rate)

    def get_trace_data(self) -> np.ndarray:
        return np.array([tr.ydata for tr in self.traces])

    def get_offsets(self) -> np.ndarray:
        return np.zeros(self.n_traces, dtype=np.int32)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        snuffle(self.traces)
