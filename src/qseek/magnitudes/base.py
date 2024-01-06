from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field
from pyrocko.trace import Trace
from typing_extensions import Self

from qseek.utils import NSL

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection, Receiver
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


class StationAmplitudes(NamedTuple):
    station_nsl: NSL
    peak: float
    noise: float
    std_noise: float

    distance_epi: float
    distance_hypo: float

    @property
    def anr(self) -> float:
        """Amplitude to noise ratio."""
        if self.noise == 0.0:
            return 0.0
        return self.peak / self.noise

    @classmethod
    def create(
        cls,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
        noise_padding: float = 3.0,
    ) -> Self:
        time_arrival = min(receiver.get_arrivals_time_window()).timestamp()
        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=time_arrival - noise_padding, inplace=False)
            for tr in traces
        ]

        peak_amp = max(np.abs(tr.ydata).max() for tr in traces)
        noise_amp = max(np.abs(tr.ydata).max() for tr in noise_traces)
        std_noise = max(np.std(tr.ydata) for tr in noise_traces)

        return cls(
            station_nsl=receiver.nsl,
            peak=peak_amp,
            noise=noise_amp,
            std_noise=std_noise,
            distance_hypo=receiver.distance_to(event),
            distance_epi=receiver.surface_distance_to(event),
        )
