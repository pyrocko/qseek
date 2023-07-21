from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from lassie.models.phase_arrival import PhaseArrival
from lassie.models.station import Stations
from lassie.utils import PhaseDescription, downsample

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from pyrocko.trace import Trace


class PickedArrival(PhaseArrival):
    provider: Literal["PickedArrival"] = "PickedArrival"


class ImageFunction(BaseModel):
    image: Literal["base"] = "base"

    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def blinding(self) -> timedelta:
        """Blinding duration for the image function. Added to padded waveforms."""
        raise NotImplementedError("must be implemented by subclass")

    def get_available_phases(self) -> tuple[str]:
        ...


@dataclass
class WaveformImage:
    image_function: ImageFunction
    phase: PhaseDescription
    weight: float
    traces: list[Trace]
    stations: Stations = Field(default_factory=lambda: Stations.model_construct())

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.delta_t

    @property
    def delta_t(self) -> float:
        return self.traces[0].deltat

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    def set_stations(self, stations: Stations) -> None:
        """Set stations from the image's available traces."""
        self.stations = stations.select_from_traces(self.traces)

    def downsample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Downsample traces in-place.
        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False.
        """
        if sampling_rate >= self.sampling_rate:
            return

        for tr in self.traces:
            if max_normalize:
                # We can use maximum here since the PhaseNet output is single-sided
                _, max_value = tr.max()
            downsample(tr, sampling_rate)

            if max_normalize:
                tr.ydata /= tr.ydata.max()
                tr.ydata *= max_value

    def get_trace_data(self) -> list[np.ndarray]:
        """Get all trace data in a list.

        Returns:
            list[np.ndarray]: List of numpy arrays.
        """
        return [tr.ydata for tr in self.traces if tr.ydata is not None]

    def get_offsets(self, reference: datetime) -> np.ndarray:
        """Get traces timing offsets to a reference time in samples.

        Args:
            reference (datetime): Reference time.

        Returns:
            np.ndarray: Integer offset towards the reference for each trace.
        """
        offsets = np.fromiter((tr.tmin for tr in self.traces), float)
        return np.round((offsets - reference.timestamp()) / self.delta_t).astype(
            np.int32
        )

    def search_phase_arrival(
        self,
        trace_idx: int,
        modelled_arrival: datetime,
        search_length_seconds: float = 5,
        threshold: float = 0.1,
    ) -> PickedArrival | None:
        """Search for a peak in all station's image functions.

        Args:
            trace_idx (int): Index of the trace.
            modelled_arrival (datetime): Time to search around.
            search_length_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.

        Returns:
            datetime | None: Time of arrival, None is none found.
        """
        raise NotImplementedError

    def search_phase_arrivals(
        self,
        modelled_arrivals: list[datetime | None],
        search_length_seconds: float = 5,
        threshold: float = 0.1,
    ) -> list[PickedArrival | None]:
        """Search for a peak in all station's image functions.

        Args:
            modelled_arrivals (list[datetime]): Time to search around.
            search_length_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.

        Returns:
            list[datetime | None]: List of arrivals, None is none found.
        """
        return [
            self.search_phase_arrival(
                idx,
                modelled_arrival,
                search_length_seconds=search_length_seconds,
                threshold=threshold,
            )
            if modelled_arrival
            else None
            for idx, modelled_arrival in zip(
                range(self.n_traces),
                modelled_arrivals,
                strict=True,
            )
        ]

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        snuffle(self.traces)
