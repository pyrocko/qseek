from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

from pydantic import BaseModel

from qseek.utils import PhaseDescription

if TYPE_CHECKING:
    import numpy as np

NSL = tuple[str, str, str]


class StationCorrections(BaseModel):
    corrections: Literal["StationCorrectionsBase"] = "StationCorrectionsBase"

    @classmethod
    def get_subclasses(cls) -> tuple[type[StationCorrections], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        subclasses = list(cls.__subclasses__())
        if len(subclasses) == 1:
            subclasses.append(cls)
        return tuple(subclasses)

    @property
    def n_stations(self) -> int:
        ...

    def get_delay(self, station_nsl: NSL, phase: PhaseDescription) -> float:
        """Get the traveltime delay for a station and phase.

        The delay is the difference between the observed and predicted traveltime.

        Args:
            station_nsl: The station NSL.
            phase: The phase description.

        Returns:
            float: The traveltime delay in seconds.
        """
        ...

    def get_delays(
        self,
        station_nsls: Iterable[NSL],
        phase: PhaseDescription,
    ) -> np.ndarray:
        """Get the traveltime delays for a set of stations and a phase.

        Args:
            station_nsls: The stations to get the delays for.
            phase: The phase to get the delays for.

        Returns:
            np.ndarray: The traveltime delays for the given stations and phase.
        """
        ...
