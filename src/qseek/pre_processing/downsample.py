from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveFloat
from pyrocko.trace import _configure_downsampling
from pyrocko.util import UnavailableDecimation, decimate_coeffs
from scipy import signal

from qseek.pre_processing.base import BatchPreProcessing, group_traces, traces_data

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.waveforms.base import WaveformBatch

logger = logging.getLogger(__name__)

# TODO make nthreads configurable
THREAD_POOL = ThreadPoolExecutor(max_workers=4)


def downsample(
    traces: list[Trace],
    delta_t: float,
    demean: bool = False,
) -> list[Trace]:
    data = traces_data(traces)
    trace_deltat = traces[0].deltat

    try:
        upscale_sratio, decimation_sequence = _configure_downsampling(
            trace_deltat, delta_t, allow_upsample_max=5
        )
    except UnavailableDecimation:
        logger.warning(
            "Cannot downsample traces to desired sampling rate. "
            "The current sampling rate is %s Hz.",
            1.0 / trace_deltat,
        )
        return traces

    if demean:
        data -= np.mean(data, axis=1, keepdims=True)

    if upscale_sratio > 1:
        data = np.repeat(data, upscale_sratio, axis=1)

    for n_decimate in decimation_sequence:
        b, a, n = decimate_coeffs(n_decimate, None, "fir-remez")
        data = signal.lfilter(b, a, data, axis=1)
        data = data[:, n // 2 :: n_decimate].copy()

    for trace, trace_data in zip(traces, data, strict=True):
        trace.ydata = trace_data
        trace.tmax = trace.tmin + (trace_data.size - 1) * delta_t
        trace.deltat = delta_t
    return traces


class Downsample(BatchPreProcessing):
    """Downsample the traces to a new sampling frequency."""

    process: Literal["downsample"] = "downsample"
    sampling_frequency: PositiveFloat = Field(
        default=100.0,
        description="The new sampling frequency in Hz.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        desired_delta_t = 1.0 / self.sampling_frequency

        def worker() -> None:
            traces = self.select_traces(batch)
            trace_groups = []
            for (delta_t, _), trace_group in group_traces(traces):
                if desired_delta_t <= delta_t:
                    logger.debug(
                        "The sampling rate of the traces is "
                        "smaller or equal to the desired rate."
                    )
                    continue
                trace_groups.append(list(trace_group))

            THREAD_POOL.map(
                downsample,
                trace_groups,
                [desired_delta_t] * len(trace_groups),
            )

        await asyncio.to_thread(worker)
        return batch
