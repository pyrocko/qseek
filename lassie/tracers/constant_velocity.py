from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PositiveFloat

from lassie.tracers.base import RayTracer
from lassie.utils import PhaseDescription, log_call

if TYPE_CHECKING:
    from lassie.models.location import Location
    from lassie.models.station import Stations
    from lassie.octree import Octree


class ConstantVelocityTracer(RayTracer):
    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    velocities: dict[PhaseDescription, PositiveFloat] = {
        "constant:P": 6000.0,
        "constant:S": 3900.0,
    }

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.velocities.keys())

    def get_traveltime_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")
        return source.distance_to(receiver) / self.velocities[phase]

    @log_call
    def get_traveltimes(
        self,
        phase: str,
        source: Octree,
        stations: Stations,
    ) -> np.ndarray:
        if phase not in self.velocities:
            raise ValueError(f"Phase {phase} is not defined.")
        distances = source.distances_stations(stations)
        return distances / self.velocities[phase]
