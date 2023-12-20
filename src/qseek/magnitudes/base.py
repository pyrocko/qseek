from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection


class StationMagnitude(NamedTuple):
    station_nsl: tuple[str, str, str]
    magnitude: float
    magnitude_error: float
    distance_epi: float
    distance_hypo: float


class EventMagnitude(BaseModel):
    magnitude: Literal["EventMagnitude"] = "EventMagnitude"

    stations: list[StationMagnitude] = []

    @cached_property
    def magnitudes(self) -> np.ndarray:
        return np.array([s.magnitude for s in self.stations])

    @computed_field
    @cached_property
    def average(self) -> float:
        """Average magnitude calculated from all stations."""
        return float(np.average(self.magnitudes))

    @computed_field
    @cached_property
    def average_weighted(self) -> float:
        """Average magnitude calculated from all stations, weighted by the
        inverse of the magnitude error."""
        return float(
            np.average(
                self.magnitudes,
                weights=[1.0 / s.magnitude_error for s in self.stations],
            )
        )

    @computed_field
    @cached_property
    def median(self) -> float:
        """Median magnitude calculated from all stations."""
        return float(np.median(self.magnitudes))

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitude], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())


class EventMagnitudeCalculator(BaseModel):
    magnitude: Literal["MagnitudeCalculator"] = "MagnitudeCalculator"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitudeCalculator], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    async def add_magnitude(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        raise NotImplementedError
