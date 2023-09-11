from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence, TypeVar

import numpy as np
from pydantic import BaseModel

from lassie.models.location import Location
from lassie.models.phase_arrival import PhaseArrival

if TYPE_CHECKING:
    from datetime import datetime

    from lassie.models.station import Stations
    from lassie.octree import Octree

_LocationType = TypeVar("_LocationType", bound=Location)


class ModelledArrival(PhaseArrival):
    tracer: Literal["ModelledArrival"] = "ModelledArrival"


class RayTracer(BaseModel):
    tracer: Literal["RayTracer"] = "RayTracer"

    async def prepare(self, octree: Octree, stations: Stations):
        ...

    def get_available_phases(self) -> tuple[str, ...]:
        ...

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        raise NotImplementedError

    def get_travel_times_locations(
        self,
        phase: str,
        source: Location,
        receivers: Sequence[Location],
    ) -> np.ndarray:
        return np.array(
            [self.get_travel_time_location(phase, source, recv) for recv in receivers]
        )

    def get_travel_times(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        """Get travel times for a phase from a source to a set of stations.

        Args:
            phase: Phase name.
            octree: Octree containing the source.
            stations: Stations to calculate travel times to.

        Returns:
            Travel times in seconds.
        """
        raise NotImplementedError

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[_LocationType],
    ) -> list[ModelledArrival | None]:
        raise NotImplementedError
