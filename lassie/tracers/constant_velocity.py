from typing import Literal

from pydantic import PositiveFloat

from lassie.models.receiver import Receiver
from lassie.octree import Node
from lassie.tracers.base import RayTracer


class ConstantVelocityTracer(RayTracer):
    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    offset: float = 0.0
    velocities: dict[str, PositiveFloat] = {
        "const.P": 5200.0,
        "const.S": 2500.0,
    }

    def get_shift(self, phase_name: str, node: Node, receiver: Receiver) -> float:
        if phase_name not in self.velocities:
            raise ValueError(f"Phase {phase_name} is not defined.")
        distance = node.distance_receiver(receiver)
        velocity = self.velocities[phase_name]
        return distance / velocity
