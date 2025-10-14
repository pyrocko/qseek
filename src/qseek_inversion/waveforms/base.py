from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from qseek.models.station import Stations

if TYPE_CHECKING:
    from pathlib import Path

    from qseek.images.images import WaveformImages


class WaveformSelection(BaseModel):
    waveforms: Literal["WaveformSelection"]

    stations: Stations = Field(
        default_factory=Stations.model_construct,
        description="Stations to use for waveform selection.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[WaveformSelection], ...]:
        return tuple(cls.__subclasses__())

    async def prepare(
        self,
        rundir: Path | None = None,
    ) -> None: ...

    async def get_images(
        self,
        window_padding: timedelta,
    ) -> list[WaveformImages]: ...
