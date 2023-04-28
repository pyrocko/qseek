from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    import numpy as np

    from lassie.models.receiver import Receiver, Receivers
    from lassie.octree import Node, Octree


class RayTracer(BaseModel):
    tracer: str

    _octree: Octree = PrivateAttr(None)
    _receivers: Receivers = PrivateAttr(None)

    def set_octree(self, octree: Octree) -> None:
        self._octree = octree

    def set_receivers(self, receivers: Receivers) -> None:
        self._receivers = receivers

    def get_available_phases(self) -> tuple[str]:
        ...

    def get_traveltime_bounds(self) -> tuple[float, float]:
        ...

    def get_traveltime(self, phase: str, node: Node, receiver: Receiver) -> float:
        ...

    def get_reveivers_traveltime(
        self, phase: str, node: Node, receivers: Receivers
    ) -> np.ndarray:
        ...
