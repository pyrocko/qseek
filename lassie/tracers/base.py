from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np

    from lassie.models.station import Station, Stations
    from lassie.octree import Node, Octree


class RayTracer(BaseModel):
    tracer: Literal["RayTracer"] = "RayTracer"

    def get_available_phases(self) -> tuple[str]:
        ...

    def get_traveltime(
        self,
        phase: str,
        node: Node,
        station: Station,
    ) -> float:
        ...

    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        ...
