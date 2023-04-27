from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from lassie.models.receiver import Receiver

if TYPE_CHECKING:
    from lassie.octree import Node


class Tracer(BaseModel):
    tracer: str

    def get_shift(self, phase_name: str, node: Node, receiver: Receiver) -> float:
        ...
