from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PositiveFloat

from qseek.pre_processing.base import BatchPreProcessing

if TYPE_CHECKING:
    from qseek.waveforms.base import WaveformBatch


class Downsample(BatchPreProcessing):
    """Downsample the traces to a new sampling frequency."""

    process: Literal["downsample"] = "downsample"
    sampling_frequency: PositiveFloat = Field(
        100.0,
        description="The new sampling frequency in Hz.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        desired_deltat = 1 / self.sampling_frequency

        def worker() -> None:
            for trace in self.select_traces(batch):
                if trace.deltat < desired_deltat:
                    trace.downsample_to(deltat=desired_deltat, allow_upsample_max=5)

        await asyncio.to_thread(worker)
        return batch
