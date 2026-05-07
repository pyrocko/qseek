from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, NamedTuple, Self

import numpy as np
from pydantic import BaseModel, Field, PositiveInt
from pyrocko import orthodrome as od
from pyrocko.trace import Trace

from qseek.base import Model
from qseek.models.location import Location
from qseek.utils import NSL

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection, Receiver
    from qseek.models.station import StationInventory
    from qseek.octree import Octree
    from qseek.waveforms.providers import WaveformProvider

KM = 1e3

PeakMeasurement = Literal["peak-to-peak", "max-amplitude", "max-amplitude-separate"]


class StationLocalMagnitude(NamedTuple):
    station: NSL
    magnitude: float
    error: float
    peak_amp: float
    distance_epi: float
    distance_hypo: float
    snr: float = 0.0


class EventMagnitude(BaseModel):
    magnitude: Literal["EventMagnitude"] = "EventMagnitude"

    average: float | None = Field(
        default=math.nan,
        description="The network's magnitude, as median of all station magnitudes.",
    )
    error: float | None = Field(
        default=math.nan,
        description="Average error of the magnitude from median absolute deviation.",
    )
    station_magnitudes: list[StationLocalMagnitude] = Field(
        default=[],
        description="The magnitudes calculated for each station.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitude], ...]:
        """Get all subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def n_observations(self) -> int:
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            "magnitude": self.average,
            "error": self.error,
        }


class EventMagnitudeCalculator(Model):
    magnitude: Literal["MagnitudeCalculator"] = "MagnitudeCalculator"

    min_stations: PositiveInt = Field(
        default=3,
        description="Minimum number of station magnitudes required to calculate "
        "the network magnitude.",
    )

    exclude_stations: list[NSL] = Field(
        default=[],
        description="List of station NSLs to exclude from magnitude calculation.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitudeCalculator], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    def has_magnitude(self, event: EventDetection) -> bool:
        """Check if the given event has a magnitude.

        Args:
            event (EventDetection): The event to check.

        Returns:
            bool: True if the event has a magnitude, False otherwise.
        """
        raise NotImplementedError

    async def get_magnitude(
        self,
        waveform_provider: WaveformProvider,
        stations: StationInventory,
        event: EventDetection,
    ) -> EventMagnitude:
        """Calculates the magnitude for the given event.

        Args:
            waveform_provider (WaveformProvider): The waveform provider.
            stations (StationInventory): The station inventory.
            event (EventDetection): The event detection object.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    async def prepare(
        self,
        octree: Octree,
        stations: StationInventory,
    ) -> None:
        """Prepare the magnitudes calculation by initializing necessary data structures.

        Args:
            octree (Octree): The octree containing seismic event data.
            stations (Stations): The stations containing seismic station data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        ...

    def csv_header(self) -> list[str]:
        """Get the CSV header for the magnitude data.

        Returns:
            list[str]: The CSV header as a list of column names.
        """
        return []


def hypo_distance_only_station_depth(location: Location, other: Location) -> float:
    """Compute 3-dimensional distance [m] to other location object.

    Ignoring elevation, this is for legacy reasons by re-implementing
    operational bad code.

    NEVER USE THIS!

    Args:
        location (Location): The location to compute the distance from.
        other (Location): The other location.

    Returns:
        float: The distance in [m].
    """
    if location._same_origin(other):
        return math.sqrt(
            (location.north_shift - other.north_shift) ** 2
            + (location.east_shift - other.east_shift) ** 2
            + (location.depth - other.effective_depth) ** 2
        )

    sx, sy, sz = od.geodetic_to_ecef(*location.effective_lat_lon, location.depth)
    ox, oy, oz = od.geodetic_to_ecef(*other.effective_lat_lon, other.effective_depth)

    return math.sqrt((sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2)


class StationAmplitudes(NamedTuple):
    station_nsl: NSL
    peak_amp: float
    noise: float
    noise_std: float

    distance_epi: float
    distance_hypo: float

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio."""
        if self.noise == 0.0:
            return 0.0
        return self.peak_amp / self.noise

    @classmethod
    def create(
        cls,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
        noise_padding: float = 1.0,
        measurement: PeakMeasurement = "max-amplitude",
        station_depth_only: bool = False,
    ) -> Self:
        if not traces:
            raise ValueError("No traces provided for amplitude calculation")

        first_arrival = min(receiver.get_arrivals_time_window()).timestamp()

        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=first_arrival - noise_padding, inplace=False)
            for tr in traces
        ]
        signal_traces = [
            tr.chop(tmin=first_arrival - noise_padding, tmax=tr.tmax, inplace=False)
            for tr in traces
        ]

        if measurement == "peak-to-peak":
            peak_amp = max(np.ptp(tr.ydata) / 2 for tr in signal_traces)
            noise_amp = max(np.ptp(tr.ydata) / 2 for tr in noise_traces)
        elif measurement == "max-amplitude":
            peak_amp = max(np.max(np.abs(tr.ydata)) for tr in signal_traces)
            noise_amp = max(np.max(np.abs(tr.ydata)) for tr in noise_traces)

        # Nobody ever really does this, custom implementation for legacy reasons
        # TODO: REMOVE!!!
        elif measurement == "max-amplitude-separate":
            if len(signal_traces) == 1:
                raise ValueError(
                    "max-amplitude-separate measurement requires at least 2 traces"
                )
            data = np.atleast_2d(np.array([tr.ydata for tr in signal_traces]))
            data_noise = np.atleast_2d(np.array([tr.ydata for tr in noise_traces]))
            peak_amp = np.linalg.norm(np.max(np.abs(data), axis=1))
            noise_amp = np.linalg.norm(np.max(np.abs(data_noise), axis=1))
        else:
            raise ValueError(f"Invalid peak measurement: {measurement}")

        noise_std = max(np.std(tr.ydata) for tr in noise_traces)

        return cls(
            station_nsl=receiver.nsl,
            peak_amp=float(peak_amp),
            noise=float(noise_amp),
            noise_std=float(noise_std),
            distance_hypo=receiver.distance_to(event)
            if not station_depth_only
            else hypo_distance_only_station_depth(receiver, event),
            distance_epi=receiver.surface_distance_to(event),
        )
