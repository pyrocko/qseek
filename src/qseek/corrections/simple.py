from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from pydantic import Field

from qseek.corrections.base import TravelTimeCorrections
from qseek.utils import NSL, NSLType, PhaseDescription

if TYPE_CHECKING:
    from qseek.octree import Node


class SimpleCorrections(TravelTimeCorrections):
    """Constant travel time corrections per station and phase.

    The station delays are added to the modeled travel times of all source
    locations.
    """

    corrections: Literal["SimpleCorrections"] = "SimpleCorrections"

    stations: dict[NSLType, dict[PhaseDescription, float]] = Field(
        default={},
        description="Travel time delay in seconds per station and phase, e.g. "
        '`{"GE.RUE.": {"cake:P": 0.12, "cake:S": 0.2}}`. Stations and phases '
        "without an entry are not corrected.",
    )

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def get_delay(
        self,
        station_nsl: NSL,
        phase: PhaseDescription,
        node: Node | None = None,
    ) -> float:
        if station_nsl not in self.stations:
            return 0.0
        if phase not in self.stations[station_nsl]:
            return 0.0
        return self.stations[station_nsl][phase]

    async def get_delays(
        self,
        station_nsls: Sequence[NSL],
        phase: PhaseDescription,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        return np.array(
            [self.get_delay(station_nsl, phase) for station_nsl in station_nsls]
        )[np.newaxis, :]
