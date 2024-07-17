from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pyrocko.trace import Trace

from qseek.utils import NSL

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.waveforms.base import WaveformBatch


class BatchPreProcessing(BaseModel):
    process: Literal["BasePreProcessing"] = "BasePreProcessing"

    stations: set[NSL] = Field(
        default=set(),
        description="List of station codes to process. E.g. ['6E.BFO', '6E.BHZ']. "
        "If empty, all stations are processed.",
    )

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

    def select_traces(self, batch: WaveformBatch) -> list[Trace]:
        """Selects traces from the given list based on the stations specified.

        Args:
            batch (WaveformBatch): The batch of traces to select from.

        Returns:
            list[Trace]: The selected traces.

        """
        if not self.stations:
            return batch.traces
        traces: list[Trace] = []
        for trace in batch.traces:
            for station in self.stations:
                if station.match(NSL.parse(trace.nslc_id)):
                    traces.append(trace)
        return traces

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


def group_traces(traces: list[Trace]) -> groupby[tuple[float, int], Trace]:
    return groupby(traces, key=lambda trace: (trace.deltat, trace.ydata.size))


def traces_data(traces: list[Trace], dtype=np.float64) -> np.ndarray:
    return np.array([trace.ydata for trace in traces], dtype=dtype)
