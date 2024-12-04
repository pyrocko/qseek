from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Sequence, TypeVar

import numpy as np
from pydantic import BaseModel

from qseek.models.location import Location

if TYPE_CHECKING:
    from pathlib import Path

    from qseek.models.station import Stations
    from qseek.octree import Node, Octree

_LocationType = TypeVar("_LocationType", bound=Location)


@dataclass
class ModelledArrival:
    phase: str
    "Name of the phase"

    time: datetime
    "Time of the arrival"

    tracer: str = ""


class RayTracer(BaseModel):
    tracer: Literal["RayTracer"] = "RayTracer"

    @classmethod
    def get_subclasses(cls) -> tuple[type[RayTracer], ...]:
        return tuple(cls.__subclasses__())

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        rundir: Path | None = None,
    ): ...

    def get_available_phases(self) -> tuple[str, ...]: ...

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

    async def get_travel_times(
        self,
        phase: str,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        """Get travel times for a phase from a source to a set of stations.

        Args:
            phase: Phase name.
            nodes: Nodes to get traveltime for.
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
