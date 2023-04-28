from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PositiveFloat

from lassie.tracers.base import RayTracer

if TYPE_CHECKING:
    from lassie.models.receiver import Receiver, Receivers
    from lassie.octree import Node


class ConstantVelocityTracer(RayTracer):
    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    offset: float = 0.0
    velocities: dict[str, PositiveFloat] = {
        "const.P": 5200.0,
        "const.S": 2500.0,
    }

    def __init__(self, **data) -> None:
        super().__init__(**data)

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.velocities.keys())

    def get_traveltime(self, phase: str, node: Node, receiver: Receiver) -> float:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")

        distance = node.distance_receiver(receiver)
        velocity = self.velocities[phase]
        return distance / velocity

    def get_receivers_traveltime(
        self,
        phase: str,
        node: Node,
        receivers: Receivers,
    ) -> np.ndarray:
        return np.fromiter(
            (self.get_traveltime(phase, node, receiver) for receiver in receivers),
            float,
        )

    def traveltime_bounds(self) -> tuple[float, float]:
        if not self._octree or not self._receivers:
            raise AttributeError("Octree and receivers must be set.")

        octree = self._octree.finest_tree()
        all_traveltimes = []
        for phase in self.get_available_phases():
            for node in octree:
                all_traveltimes.append(
                    self.get_receivers_traveltime(phase, node, self._receivers)
                )

        all_traveltimes = np.ndarray(all_traveltimes)
        return all_traveltimes.min(), all_traveltimes.max()
