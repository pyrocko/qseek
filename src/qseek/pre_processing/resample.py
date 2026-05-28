from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, PositiveFloat
from pyrocko.trace import _configure_downsampling
from pyrocko.util import UnavailableDecimation, decimate_coeffs
from scipy import signal

from qseek.pre_processing.base import BatchPreProcessing, group_traces, traces_data
from qseek.utils import to_threadpool

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.waveforms.base import WaveformBatch

logger = logging.getLogger(__name__)


def resample(
    traces: list[Trace],
    delta_t: float,
    demean: bool = False,
) -> list[Trace]:
    """Resample traces using polyphase filtering to exactly match the target rate.

    Args:
        traces: The traces to resample.
        delta_t: The new sampling interval in seconds.
        demean: Whether to remove the mean before resampling. Defaults to False.

    Returns:
        list[Trace]: The resampled traces.
    """
    if len({tr.deltat for tr in traces}) > 1:
        raise ValueError("Traces have different sampling rates, cannot resample.")

    in_deltat = traces[0].deltat
    data = traces_data(traces)
    if demean:
        data -= np.mean(data, axis=1, keepdims=True)

    ratio = Fraction(in_deltat / delta_t).limit_denominator(1000)
    data_resampled = signal.resample_poly(
        data,
        ratio.numerator,
        ratio.denominator,
        window=("kaiser", 10),
        axis=1,
    )
    out_deltat = in_deltat * ratio.denominator / ratio.numerator

    for trace, trace_data in zip(traces, data_resampled, strict=True):
        trace.deltat = out_deltat
        trace.set_ydata(trace_data)
    return traces


def downsample(
    traces: list[Trace],
    delta_t: float,
    demean: bool = False,
) -> list[Trace]:
    """Downsample the traces, the decimation has to be an integer factor.

    Args:
        traces (list[Trace]): The traces to downsample.
        delta_t (float): The new sampling interval in seconds.
        demean (bool, optional): Whether to remove the mean from the traces.
        Defaults to False.

    Returns:
        list[Trace]: The downsampled traces.
    """
    if len({tr.deltat for tr in traces}) != 1:
        raise ValueError("Traces have different sampling rates, cannot downsample.")

    traces_deltat = traces[0].deltat
    try:
        upscale_sratio, decimation_sequence = _configure_downsampling(
            traces_deltat,
            delta_t,
            allow_upsample_max=5,
        )
    except UnavailableDecimation:
        logger.warning(
            "Cannot downsample traces to desired sampling rate. "
            "The current sampling rate is %s Hz.",
            1.0 / traces_deltat,
        )
        return traces

    data = traces_data(traces)
    if demean:
        data -= np.mean(data, axis=1, keepdims=True)

    if upscale_sratio > 1:
        data = np.repeat(data, upscale_sratio, axis=1)

    for n_decimate in decimation_sequence:
        b, a, n = decimate_coeffs(n_decimate, None, "fir-remez")
        data = signal.sosfilt(signal.tf2sos(b, a), data, axis=1)
        data = data[:, n // 2 :: n_decimate].copy()

    for trace, trace_data in zip(traces, data, strict=True):
        trace.deltat = delta_t
        trace.set_ydata(trace_data)
    return traces


class Downsample(BatchPreProcessing):
    """Downsample the traces to a new sampling frequency."""

    process: Literal["downsample"] = "downsample"

    sampling_frequency: PositiveFloat = Field(
        default=100.0,
        description="The new sampling frequency in Hz.",
    )
    n_threads: int = Field(
        default=8,
        description="The number of threads to use for downsampling.",
    )

    _thread_pool: ThreadPoolExecutor

    def model_post_init(self, context: Any):
        self._thread_pool = ThreadPoolExecutor(max_workers=self.n_threads)

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        desired_delta_t = 1.0 / self.sampling_frequency
        traces = self.filter_traces(batch)
        groups = [
            list(trace_group)
            for (delta_t, _), trace_group in group_traces(traces)
            if desired_delta_t < delta_t
        ]
        await asyncio.gather(
            *(
                to_threadpool(self._thread_pool, downsample, group, desired_delta_t)
                for group in groups
            )
        )
        return batch


class Resample(BatchPreProcessing):
    """Resample the traces to a new sampling frequency."""

    process: Literal["resample"] = "resample"

    sampling_frequency: PositiveFloat = Field(
        default=100.0,
        description="The new sampling frequency in Hz.",
    )
    n_threads: int = Field(
        default=8,
        description="The number of threads to use for resampling.",
    )

    _thread_pool: ThreadPoolExecutor

    def model_post_init(self, context: Any):
        self._thread_pool = ThreadPoolExecutor(max_workers=self.n_threads)

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        new_delta_t = 1.0 / self.sampling_frequency
        traces = self.filter_traces(batch)
        groups = [
            list(trace_group)
            for (delta_t, _), trace_group in group_traces(traces)
            if delta_t != new_delta_t
        ]
        await asyncio.gather(
            *(
                to_threadpool(self._thread_pool, resample, group, new_delta_t)
                for group in groups
            )
        )
        return batch
