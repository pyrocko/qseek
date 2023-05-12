from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PositiveFloat

from lassie.tracers.base import RayTracer
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from lassie.models.station import Station, Stations
    from lassie.octree import Node, Octree


class ConstantVelocityTracer(RayTracer):
    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    velocities: dict[PhaseDescription, PositiveFloat] = {
        "constant:P": 6000.0,
        "constant:S": 3900.0,
    }

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.velocities.keys())

    def get_traveltime(self, phase: str, node: Node, station: Station) -> float:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")
        return node.distance_station(station) / self.velocities[phase]

    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")
        distances = octree.distances_stations(stations)
        return distances / self.velocities[phase]
