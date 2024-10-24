from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveFloat, field_validator
from scipy import signal

from qseek.pre_processing.base import BatchPreProcessing, group_traces, traces_data
from qseek.utils import Range

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)


@lru_cache
def butter_sos(
    N: int,  # noqa: N803
    Wn: float | tuple[float, float],  # noqa: N803
    btype: Literal["lowpass", "highpass", "bandpass"],
    fs: float,
    dtype: np.dtype = float,
) -> np.ndarray:
    return signal.butter(
        N=N,
        Wn=Wn,
        btype=btype,
        fs=fs,
        output="sos",
    ).astype(dtype)


def _sos_filter(
    traces: list[Trace],
    sos: np.ndarray,
    demean: bool,
    zero_phase: bool,
) -> list[Trace]:
    data = traces_data(traces)
    if demean:
        data -= np.mean(data, axis=1, keepdims=True)

    if zero_phase:
        data = signal.sosfiltfilt(sos, data, axis=1)
    else:
        data = signal.sosfilt(sos, data, axis=1)

    for trace, ydata in zip(traces, data, strict=True):
        trace.set_ydata(ydata)
    return traces


class Bandpass(BatchPreProcessing):
    """Apply a bandpass filter to the traces."""

    process: Literal["bandpass"] = "bandpass"

    corners: int = Field(
        default=4,
        ge=2,
        description="Number of corners for the filter.",
    )
    bandpass: Range = Field(
        default=Range(0.5, 30.0),
        description="The highpass frequency.",
    )
    demean: bool = Field(
        default=True,
        description="If True, demean the trace before filtering.",
    )

    @field_validator("bandpass")
    @classmethod
    def _check_bandpass(cls, value) -> Range:
        if value[0] >= value[1]:
            raise ValueError(
                "The lowpass frequency must be smaller than the highpass frequency."
            )
        return value

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            traces = self.select_traces(batch)
            for (deltat, _), trace_group in group_traces(traces):
                sampling_rate = 1.0 / deltat
                if self.bandpass.max >= sampling_rate / 2:
                    logger.debug(
                        "Highpass frequency is higher than Nyquist frequency %s. "
                        "No filtering is applied.",
                        sampling_rate / 2,
                    )
                    continue
                sos = butter_sos(
                    N=self.corners,
                    Wn=self.bandpass,
                    btype="bandpass",
                    fs=1.0 / deltat,
                )
                _sos_filter(list(trace_group), sos, demean=self.demean, zero_phase=True)

        await asyncio.to_thread(worker)
        return batch


class Highpass(BatchPreProcessing):
    """Apply a highpass filter to the traces."""

    process: Literal["highpass"] = "highpass"
    corners: int = Field(
        default=4,
        ge=2,
        description="Number of corners for the filter.",
    )
    frequency: PositiveFloat = Field(
        default=0.1,
        description="The highpass frequency.",
    )
    demean: bool = Field(
        default=True,
        description="If True, demean the trace before filtering.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            traces = self.select_traces(batch)
            for (deltat, _), trace_group in group_traces(traces):
                sampling_rate = 1.0 / deltat
                if self.frequency >= sampling_rate / 2:
                    logger.debug(
                        "Highpass frequency is higher than Nyquist frequency %s. "
                        "No filtering is applied.",
                        sampling_rate / 2,
                    )
                    continue
                sos = butter_sos(
                    N=self.corners,
                    Wn=self.frequency,
                    btype="highpass",
                    fs=sampling_rate,
                )
                _sos_filter(list(trace_group), sos, demean=self.demean, zero_phase=True)

        await asyncio.to_thread(worker)
        return batch


class Lowpass(BatchPreProcessing):
    """Apply a lowpass filter to the traces."""

    process: Literal["lowpass"] = "lowpass"
    corners: int = Field(
        default=4,
        ge=2,
        description="Number of corners for the filter.",
    )
    frequency: PositiveFloat = Field(
        default=0.1,
        description="The highpass frequency.",
    )
    demean: bool = Field(
        default=True,
        description="If True, demean the trace before filtering.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            traces = self.select_traces(batch)
            for (deltat, _), trace_group in group_traces(traces):
                sampling_rate = 1.0 / deltat
                if self.frequency >= sampling_rate / 2:
                    logger.debug(
                        "Lowpass frequency is higher than Nyquist frequency. "
                        "No filtering is applied."
                    )
                    continue
                sos = butter_sos(
                    N=self.corners,
                    Wn=self.frequency,
                    btype="lowpass",
                    fs=sampling_rate,
                )
                _sos_filter(list(trace_group), sos, demean=self.demean, zero_phase=True)

        await asyncio.to_thread(worker)
        return batch
