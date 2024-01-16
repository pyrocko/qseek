from typing import Iterable, Literal

import numpy as np

from qseek.corrections.base import StationCorrections
from qseek.utils import NSL, PhaseDescription


class SimpleCorrections(StationCorrections):
    corrections: Literal["SimpleCorrections"] = "SimpleCorrections"

    stations: dict[NSL, dict[PhaseDescription, float]] = {}

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def get_delay(self, station_nsl: NSL, phase: PhaseDescription) -> float:
        if station_nsl not in self.stations:
            return 0.0
        if phase not in self.stations[station_nsl]:
            return 0.0
        return self.stations[station_nsl][phase]

    def get_delays(
        self,
        station_nsls: Iterable[NSL],
        phase: PhaseDescription,
    ) -> np.ndarray:
        return np.array(
            [self.get_delay(station_nsl, phase) for station_nsl in station_nsls]
        )
