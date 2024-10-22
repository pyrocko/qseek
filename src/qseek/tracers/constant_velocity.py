from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import Field, PositiveFloat

from qseek.octree import distances_stations
from qseek.tracers.base import ModelledArrival, RayTracer
from qseek.utils import PhaseDescription

if TYPE_CHECKING:
    import numpy as np

    from qseek.models.location import Location
    from qseek.models.station import Stations
    from qseek.octree import Node


class ConstantVelocityTracer(RayTracer):
    """Travel time tracer with a constant velocity."""

    tracer: Literal["ConstantVelocityTracer"] = "ConstantVelocityTracer"
    phase: PhaseDescription = Field(
        default="constant:P",
        description="Name of the phase.",
    )
    velocity: PositiveFloat = Field(
        default=5000.0,
        description="Constant velocity of the phase in m/s.",
    )

    def get_available_phases(self) -> tuple[str, ...]:
        return (self.phase,)

    def _check_phase(self, phase: PhaseDescription) -> None:
        if phase != self.phase:
            raise ValueError(f"Phase {phase} is not defined.")

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        self._check_phase(phase)
        return source.distance_to(receiver) / self.velocity

    async def get_travel_times(
        self,
        phase: str,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        self._check_phase(phase)

        distances = distances_stations(nodes, stations)
        return distances / self.velocity

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival]:
        self._check_phase(phase)

        traveltimes = self.get_travel_times_locations(
            phase,
            source=source,
            receivers=receivers,
        )
        arrivals = []
        for traveltime in traveltimes:
            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = ModelledArrival(
                phase=phase,
                time=arrivaltime,
            )
            arrivals.append(arrival)
        return arrivals
