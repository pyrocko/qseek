from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection
    from qseek.models.station import Stations
    from qseek.octree import Octree


class EventMagnitude(BaseModel):
    magnitude: Literal["EventMagnitude"] = "EventMagnitude"

    average: float = Field(
        default=0.0,
        description="Average local magnitude.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of local magnitude.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitude], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def csv_row(self) -> dict[str, float]:
        return {
            "magnitude": self.average,
            "error": self.error,
        }


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
        """
        Adds a magnitude to the squirrel for the given event.

        Args:
            squirrel (Squirrel): The squirrel object to add the magnitude to.
            event (EventDetection): The event detection object.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
    ) -> None:
        """
        Prepare the magnitudes calculation by initializing necessary data structures.

        Args:
            octree (Octree): The octree containing seismic event data.
            stations (Stations): The stations containing seismic station data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
