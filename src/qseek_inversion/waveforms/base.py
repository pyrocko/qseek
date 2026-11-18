from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from pathlib import Path

    from qseek.images.base import WaveformImages
    from qseek.models.station import StationInventory
    from qseek.search import Search


class WaveformSelection(BaseModel):
    waveforms: Literal["WaveformSelection"]

    _stations: StationInventory = PrivateAttr()

    @classmethod
    def get_subclasses(cls) -> tuple[type[WaveformSelection], ...]:
        return tuple(cls.__subclasses__())

    async def prepare(
        self,
        search: Search,
        import_rundir: Path,
        rundir: Path | None = None,
    ) -> None:
        """Select the events and waveforms of the inversion.

        Args:
            search (Search): The search configuration of the imported run.
            import_rundir (Path): The run directory of the imported run.
            rundir (Path | None, optional): Directory to export the selection to.
                Defaults to None.
        """
        self._stations = search.stations

    def get_stations(self) -> StationInventory:
        """Stations of the selected waveforms."""
        return self._stations

    def get_blinding(self) -> timedelta:
        """Blinding of the image function at the edges of the waveforms."""
        return timedelta(seconds=0.0)

    async def get_images(
        self,
        window_padding: timedelta,
    ) -> list[WaveformImages]:
        raise NotImplementedError
