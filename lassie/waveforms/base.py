from __future__ import annotations

import logging
from asyncio import Queue
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, AsyncIterator, Literal

import numpy as np
from pydantic import BaseModel, PrivateAttr
from pyrocko.trace import Trace

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from lassie.models.station import Stations

logger = logging.getLogger(__name__)


@dataclass
class WaveformBatch:
    traces: list[Trace]
    start_time: datetime
    end_time: datetime
    i_batch: int
    n_batches: int = 0

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def cumulative_duration(self) -> timedelta:
        """Cumulative duration of the traces in the batch.

        Returns:
            timedelta: Cumulative duration of the traces in the batch.
        """
        seconds = 0.0
        for tr in self.traces:
            seconds += tr.tmax - tr.tmin
        return timedelta(seconds=seconds)

    @property
    def cumulative_bytes(self) -> int:
        return sum(tr.ydata.nbytes for tr in self.traces)

    def is_empty(self) -> bool:
        """Check if the batch is empty.

        Returns:
            bool: True if the batch is empty, False otherwise.
        """
        return not bool(self.traces)

    def clean_traces(self) -> None:
        """Remove empty or bad traces."""
        for tr in self.traces.copy():
            if not tr.ydata.size or not np.all(np.isfinite(tr.ydata)):
                logger.warning("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)

    def log_str(self) -> str:
        """Log the batch."""
        return f"{self.i_batch+1}/{self.n_batches or '?'} {self.start_time}"


class WaveformProvider(BaseModel):
    provider: Literal["WaveformProvider"] = "WaveformProvider"

    _queue: Queue[WaveformBatch | None] = PrivateAttr(default_factory=lambda: Queue())

    def get_squirrel(self) -> Squirrel:
        raise NotImplementedError

    def prepare(self, stations: Stations) -> None:
        ...

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
    ) -> AsyncIterator[WaveformBatch]:
        yield
        raise NotImplementedError
