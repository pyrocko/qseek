from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PositiveFloat, field_validator

from qseek.pre_processing.base import BatchPreProcessing

if TYPE_CHECKING:
    from qseek.waveforms.base import WaveformBatch


class Bandpass(BatchPreProcessing):
    """Apply a bandpass filter to the traces."""

    process: Literal["bandpass"] = "bandpass"

    corners: int = Field(
        4,
        ge=2,
        description="Number of corners for the filter.",
    )
    bandpass: tuple[PositiveFloat, PositiveFloat] = Field(
        0.1,
        description="The highpass frequency.",
    )
    demean: bool = Field(
        True,
        description="If True, demean the trace before filtering.",
    )

    @field_validator("bandpass")
    @classmethod
    def _check_bandpass(cls, value) -> tuple[PositiveFloat, PositiveFloat]:
        if value[0] >= value[1]:
            raise ValueError(
                "The lowpass frequency must be smaller than the highpass frequency."
            )
        return value

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            for trace in self.select_traces(batch):
                trace.bandpass(
                    order=self.corners,
                    corner_hp=self.bandpass[0],
                    corner_lp=self.bandpass[1],
                    demean=self.demean,
                )

        await asyncio.to_thread(worker)
        return batch


class Highpass(BatchPreProcessing):
    """Apply a highpass filter to the traces."""

    process: Literal["highpass"] = "highpass"
    corners: int = Field(
        4,
        ge=2,
        description="Number of corners for the filter.",
    )
    frequency: PositiveFloat = Field(
        0.1,
        description="The highpass frequency.",
    )
    demean: bool = Field(
        True,
        description="If True, demean the trace before filtering.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            for trace in self.select_traces(batch):
                trace.highpass(
                    order=self.corners,
                    corner=self.frequency,
                    demean=self.demean,
                )

        await asyncio.to_thread(worker)
        return batch


class Lowpass(BatchPreProcessing):
    """Apply a lowpass filter to the traces."""

    process: Literal["lowpass"] = "lowpass"
    corners: int = Field(
        4,
        ge=2,
        description="Number of corners for the filter.",
    )
    frequency: PositiveFloat = Field(
        0.1,
        description="The highpass frequency.",
    )
    demean: bool = Field(
        True,
        description="If True, demean the trace before filtering.",
    )

    async def process_batch(self, batch: WaveformBatch) -> WaveformBatch:
        def worker() -> None:
            for trace in self.select_traces(batch):
                trace.lowpass(
                    order=self.corners,
                    corner=self.frequency,
                    demean=self.demean,
                )

        await asyncio.to_thread(worker)
        return batch
