from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.plot.base import LassieFigure
    from qseek.search import Search


class PlotManager(BaseModel):
    _search: Search = PrivateAttr()

    def set_search(self, search: Search) -> None:
        self._search = search

    def get_event_figures(self, event: EventDetection) -> list[LassieFigure]:
        raise NotImplementedError

    @classmethod
    def from_rundir(cls, rundir: Path) -> PlotManager:
        manager = cls()
        manager.set_search(Search.load_rundir(rundir))
        return manager
