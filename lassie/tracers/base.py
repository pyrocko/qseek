from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from lassie.models.receiver import Receiver, Receivers
    from lassie.octree import Node, Octree


class RayTracer(BaseModel):
    tracer: str

    _octree: Octree = PrivateAttr(None)
    _receivers: Receivers = PrivateAttr(None)

    def available_phase(self) -> tuple[str]:
        ...

    def get_traveltime(self, phase: str, node: Node, receiver: Receiver) -> float:
        ...

    def set_octree(self, octree: Octree) -> None:
        self._octree = octree

    def set_receivers(self, receivers: Receivers) -> None:
        self._receivers = receivers

    def prepare(self) -> None:
        ...
