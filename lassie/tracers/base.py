from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    import numpy as np

    from lassie.models.stations import Station, Stations
    from lassie.octree import Node, Octree


class RayTracer(BaseModel):
    tracer: str

    _octree: Octree = PrivateAttr(None)
    _stations: Stations = PrivateAttr(None)

    def set_octree(self, octree: Octree) -> None:
        self._octree = octree

    def set_stations(self, stations: Stations) -> None:
        self._stations = stations

    def get_available_phases(self) -> tuple[str]:
        ...

    def get_traveltime_bounds(self) -> tuple[float, float]:
        ...

    def get_traveltime(self, phase: str, node: Node, station: Station) -> float:
        ...

    def get_traveltimes(self, phase: str, node: Node, stations: Stations) -> np.ndarray:
        ...
