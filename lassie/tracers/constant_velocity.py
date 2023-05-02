from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PositiveFloat

from lassie.tracers.base import RayTracer

if TYPE_CHECKING:
    from lassie.models.station import Station, Stations
    from lassie.octree import Node


class ConstantVelocityTracer(RayTracer):
    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    offset: float = 0.0
    velocities: dict[str, PositiveFloat] = {
        "constant.P": 5200.0,
        "constant.S": 2500.0,
    }

    def __init__(self, **data) -> None:
        super().__init__(**data)

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.velocities.keys())

    def get_traveltime(self, phase: str, node: Node, station: Station) -> float:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")

        distance = node.distance_station(station)
        velocity = self.velocities[phase]
        return distance / velocity

    def get_traveltimes(
        self,
        phase: str,
        node: Node,
        stations: Stations,
    ) -> np.ndarray:
        return np.fromiter(
            (self.get_traveltime(phase, node, station) for station in stations), float
        )

    def traveltime_bounds(self) -> tuple[float, float]:
        if not self._octree or not self._stations:
            raise AttributeError("Octree and receivers must be set.")

        octree = self._octree.finest_tree()
        all_traveltimes = []
        for phase in self.get_available_phases():
            for node in octree:
                all_traveltimes.append(
                    self.get_traveltimes(phase, node, self._stations)
                )

        all_traveltimes = np.ndarray(all_traveltimes)
        return all_traveltimes.min(), all_traveltimes.max()
