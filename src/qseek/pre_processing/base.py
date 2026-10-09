from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, PositiveInt, PrivateAttr, field_validator
from pyrocko.trace import Trace

from qseek.utils import NSL, NSLType

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.waveforms.base import WaveformBatch

# Minimum samples per chunk of traces processed in one thread
MIN_CHUNK_SAMPLES = 200_000


class BatchPreProcessing(BaseModel):
    process: Literal["BasePreProcessing"] = "BasePreProcessing"

    stations: set[NSLType] = Field(
        default=set(),
        description="List of station codes to process. E.g. ['6E.BFO', '6E.BHZ']. "
        "If empty, all stations are processed.",
    )
    n_threads: PositiveInt = Field(
        default=8,
        description="The number of threads processing the traces in parallel. "
        "Ignored by the DeepDenoiser.",
    )

    _thread_pool: ThreadPoolExecutor | None = PrivateAttr(None)

    @field_validator("stations")
    @classmethod
    def validate_stations(cls, v) -> set[NSL]:
        stations = set()
        for station in v:
            stations.add(NSL.parse(station))
        return stations

    @classmethod
    def get_subclasses(cls) -> tuple[type[BatchPreProcessing], ...]:
        """Returns a tuple of all the subclasses of BasePreProcessing."""
        return tuple(cls.__subclasses__())

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """The thread pool of the module, created on first use."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.n_threads)
        return self._thread_pool

    def filter_traces(self, batch: WaveformBatch) -> list[Trace]:
        """Selects traces from the given list based on the stations specified.

        Args:
            batch (WaveformBatch): The batch of traces to select from.

        Returns:
            list[Trace]: The selected traces.

        """
        if not self.stations:
            return batch.traces

        return [
            trace
            for trace in batch.traces
            if any(station.match(NSL(*trace.nslc_id[:3])) for station in self.stations)
        ]

    async def prepare(self) -> None:
        """Prepare the pre-processing module."""
        pass

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        """Process a list of traces.

        Args:
            batch (WaveformBatch): The batch of traces to process.

        Returns:
            list[Trace]: The processed list of traces.
        """
        raise NotImplementedError


def _trace_group_key(trace: Trace) -> tuple[float, int]:
    return (trace.deltat, trace.ydata.size)


def group_traces(traces: list[Trace]) -> groupby[tuple[float, int], Trace]:
    return groupby(sorted(traces, key=_trace_group_key), key=_trace_group_key)


def split_traces(
    traces: list[Trace],
    n_chunks: int,
    min_samples: int = MIN_CHUNK_SAMPLES,
) -> list[list[Trace]]:
    """Split the traces into at most n_chunks chunks of about equal size.

    The pre-processing treats every trace on its own, the chunks of a group from
    :func:`group_traces` can be processed in parallel threads. A chunk holds at
    least min_samples samples, smaller chunks cost more threading than they save.
    """
    if not traces:
        return []
    min_size = math.ceil(min_samples / max(traces[0].ydata.size, 1))
    size = max(math.ceil(len(traces) / n_chunks), min_size)
    return [traces[i : i + size] for i in range(0, len(traces), size)]


def traces_data(traces: list[Trace], dtype=np.float32) -> np.ndarray:
    data_sizes = {trace.ydata.size for trace in traces}
    if len(data_sizes) != 1:
        raise ValueError("Traces have different number of samples.")
    out = np.empty((len(traces), data_sizes.pop()), dtype=dtype)
    for itr, trace in enumerate(traces):
        np.copyto(out[itr], trace.ydata, casting="unsafe")
    return out
