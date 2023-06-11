from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from pydantic import PositiveFloat

from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.utils import PhaseDescription, log_call

if TYPE_CHECKING:
    from lassie.models.location import Location
    from lassie.models.station import Stations
    from lassie.octree import Octree


class ConstantVelocityArrival(ModelledArrival):
    tracer: Literal["ConstantVelocityArrival"] = "ConstantVelocityArrival"
    phase: str


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

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ConstantVelocityArrival]:
        traveltimes = self.get_traveltimes_locations(
            phase,
            source=source,
            receivers=receivers,
        )
        arrivals = []
        for traveltime in traveltimes:
            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = ConstantVelocityArrival(time=arrivaltime, phase=phase)
            arrivals.append(arrival)
        return arrivals

    def get_velocity_max(self) -> float:
        return max(self.velocities.values())
