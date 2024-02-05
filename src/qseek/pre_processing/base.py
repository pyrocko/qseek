from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

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
        """
        Returns a tuple of all the subclasses of BasePreProcessing.
        """
        return tuple(cls.__subclasses__())

    def select_traces(self, batch: WaveformBatch) -> list[Trace]:
        """
        Selects traces from the given list based on the stations specified.

        Args:
            traces (list[Trace]): The list of traces to select from.

        Returns:
            list[Trace]: The selected traces.

        """
        if not self.stations:
            return batch.traces
        return [
            trace
            for trace in batch.traces
            if NSL.parse(trace.nslc_id).station in self.stations
        ]

    async def prepare(self) -> None:
        """
        Prepare the pre-processing module.
        """
        pass

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        """
        Process a list of traces.

        Args:
            traces (list[Trace]): The list of traces to be processed.

        Returns:
            list[Trace]: The processed list of traces.
        """
        raise NotImplementedError
