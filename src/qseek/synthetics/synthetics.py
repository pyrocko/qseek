from __future__ import annotations

import logging
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from pyrocko.io import save
from pyrocko.model.event import Event, dump_events
from pyrocko.trace import Trace

from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.tracers.base import RayTracer
from qseek.utils import NSL, PhaseDescription

logger = logging.getLogger(__name__)


def _gaussian_function(sigma: int = 10):
    length = sigma * 10
    onset = length // 2
    x = np.arange(length)
    return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


class Receiver(Station):
    arrivals: dict[PhaseDescription, float] = Field(
        default_factory=dict,
        description="Dictionary of phase names to travel times (in seconds).",
    )

    def add_arrival(self, phase: PhaseDescription, travel_time: float) -> None:
        if np.isfinite(travel_time):
            self.arrivals[phase] = travel_time

    def as_station(self) -> Station:
        return Station(**self.model_dump())


class SyntheticEvent(Location):
    origin_time: datetime = Field(
        ...,
        description="Origin time of the synthetic event.",
    )
    magnitude: float = Field(
        default=2.0,
        description="Magnitude of the synthetic event.",
    )
    receivers: dict[NSL, Receiver] = Field(
        default_factory=dict,
        description="List of stations receiving this event.",
    )

    def add_receiver(self, station: Station) -> None:
        """Add a receiver for the given station.

        Args:
            station (Station): Station object to add as receiver.
        """
        nsl = station.nsl
        if nsl in self.receivers:
            raise ValueError(f"Receiver for station {nsl} already exists.")

        receiver = Receiver(**station.model_dump())
        self.receivers[nsl] = receiver

    def get_receiver(self, station: Station) -> Receiver:
        """Get or create a receiver for the given station.

        If the receiver does not exist, it is created and added to
        the receivers dictionary.

        Args:
            station (Station): Station object or NSL identifier.

        Raises:
            ValueError: If the station is not found in receivers.

        Returns:
            Receiver: Receiver object for the station.
        """
        nsl = station.nsl
        if nsl not in self.receivers:
            self.add_receiver(station)
        return self.receivers[nsl]

    def get_receivers(self) -> list[Receiver]:
        """Get the list of receivers for this event.

        Returns:
            list[Receiver]: List of receivers.
        """
        return list(self.receivers.values())

    def get_travel_time(self, phase: PhaseDescription, nsl: NSL) -> datetime | None:
        """Get the travel time for a given phase and station.

        Args:
            phase (PhaseDescription): Phase name.
            nsl (NSL): NSL identifier of the station.

        Returns:
            float | None: Travel time in seconds, or None if not found.
        """
        receiver = self.receivers.get(nsl)
        if receiver is None:
            return None
        travel_time = receiver.arrivals.get(phase)
        if travel_time is None:
            return None
        return self.origin_time + timedelta(seconds=travel_time)

    def first_arrival(self) -> datetime | None:
        """Get the time of the first arrival across all receivers and phases.

        Returns:
            datetime | None: Time of the first arrival, or None if no arrivals exist.
        """
        arrivals = chain(*((rcv.arrivals.values()) for rcv in self.receivers.values()))
        first_arrival = min(arrivals, default=None)
        return self.origin_time + timedelta(seconds=first_arrival or 0.0)

    def last_arrival(self) -> datetime | None:
        """Get the time of the first arrival across all receivers and phases.

        Returns:
            datetime | None: Time of the first arrival, or None if no arrivals exist.
        """
        arrivals = chain(*((rcv.arrivals.values()) for rcv in self.receivers.values()))
        last_arrival = min(arrivals, default=None)
        return self.origin_time + timedelta(seconds=last_arrival or 0.0)

    def available_phases(self) -> set[PhaseDescription]:
        """Get the set of available phases for this event.

        Returns:
            set[PhaseDescription]: Set of phase names.
        """
        phases = set()
        for receiver in self.get_receivers():
            phases.update(receiver.arrivals.keys())
        return phases

    def as_pyrocko_event(self) -> Event:
        return Event(
            name=self.origin_time.isoformat(),
            lat=self.effective_lat,
            lon=self.effective_lon,
            depth=self.effective_depth,
            time=self.origin_time.timestamp(),
            magnitude=self.magnitude,
            extras={"n_receivers": len(self.receivers)},
        )

    def as_csv(self) -> str:
        lat, lon = self.effective_lat_lon
        depth = self.effective_depth
        return (
            f"{self.origin_time},{lat:6f},{lon:6f},{depth:6f},{self.magnitude:.2f},"
            f"{self.as_wkt()}"
        )


class SyntheticEventCatalog(BaseModel):
    events: list[SyntheticEvent] = Field(
        default_factory=list,
        description="List of synthetic events.",
    )

    def add_event(
        self,
        event: SyntheticEvent,
        stations: Stations,
        ray_tracer: RayTracer,
    ) -> SyntheticEvent:
        """Add a synthetic event and compute arrivals at stations.

        Args:
            event (SyntheticEvent): Event to add.
            stations (Stations): Stations to compute arrivals for.
            ray_tracer (RayTracer): Ray tracer to use for travel time computation.
        """
        for phase in ray_tracer.get_available_phases():
            for station in stations:
                travel_time = ray_tracer.get_travel_time_location(
                    phase=phase,
                    source=event,
                    receiver=station,
                )
                receiver = event.get_receiver(station)
                receiver.add_arrival(phase, travel_time)
        self.events.append(event)
        logger.debug(
            "added synthetic event at %s with %d receivers",
            event.origin_time,
            len(event.receivers),
        )
        return event

    def get_traces(
        self,
        start_time: datetime,
        end_time: datetime,
        phase: PhaseDescription | None = None,
        sampling_rate: float = 100.0,
        sigma_samples: int = 10,
    ) -> list[Trace]:
        if not self.events:
            raise ValueError("No events in the catalog.")

        events = [ev for ev in self.events if start_time <= ev.origin_time <= end_time]
        nsls = {rcv.nsl for ev in events for rcv in ev.get_receivers()}
        n_samples = int((end_time - start_time).total_seconds() * sampling_rate)

        def filter_arrival_times(travel_time: datetime | None) -> bool:
            if travel_time is None:
                return False
            return start_time <= travel_time < end_time

        traces = []
        for nsl in nsls:
            arrivals = filter(
                filter_arrival_times, [ev.get_travel_time(phase, nsl) for ev in events]
            )
            arrival_times: list[datetime] = list(arrivals)  # type: ignore
            if not arrival_times:
                continue

            data = np.zeros(n_samples, dtype=np.float32)
            arrival_time_idx = [
                int((arrival - start_time).total_seconds() * sampling_rate)
                for arrival in arrival_times
            ]
            data[arrival_time_idx] = 1.0  # Simple spike for arrivals
            gaussian = _gaussian_function(sigma=sigma_samples)
            data = np.convolve(data, gaussian, mode="same")
            trace = Trace(
                network=nsl.network,
                station=nsl.station,
                location=nsl.location,
                channel=phase[-1],
                tmin=start_time.timestamp(),
                deltat=1.0 / sampling_rate,
                ydata=data,
            )
            traces.append(trace)

        return traces

    def get_available_phases(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> set[PhaseDescription]:
        phases = set()
        span_start, span_end = self.get_time_span()
        start_time = start_time or span_start
        end_time = end_time or span_end
        events = [ev for ev in self.events if start_time <= ev.origin_time <= end_time]
        for event in events:
            phases.update(event.available_phases())
        return phases

    def get_all_traces(
        self,
        start_time: datetime,
        end_time: datetime,
        sampling_rate: float = 100.0,
        sigma_samples: int = 10,
    ) -> list[Trace]:
        phases = self.get_available_phases(
            start_time=start_time,
            end_time=end_time,
        )

        all_traces = []
        for phase in phases:
            traces = self.get_traces(
                phase=phase,
                start_time=start_time,
                end_time=end_time,
                sampling_rate=sampling_rate,
                sigma_samples=sigma_samples,
            )
            all_traces.extend(traces)
        return all_traces

    def export_csv(self, path: Path) -> None:
        """Export the synthetic event catalog to a CSV string.

        Args:
            path (Path): Path to export the CSV file to.

        Returns:
            str: CSV representation of the catalog.
        """
        csv_catalog = path / "catalog.csv"
        header = "origin_time,latitude,longitude,depth,magnitude,WKT_geom"
        lines = [header]
        for event in self.events:
            lines.append(event.as_csv())
        csv_catalog.write_text("\n".join(lines))

        stations = self.get_stations()
        stations.export_csv(path / "stations.csv")
        logger.info("exported CSV models to %s", path)

    def export_pyrocko(self, path: Path) -> None:
        """Export the synthetic event catalog to a Pyrocko YAML events file.

        Args:
            path (Path): Path to export the events file to.
        """
        pyrocko_events = [event.as_pyrocko_event() for event in self.events]
        dump_events(pyrocko_events, str(path / "pyrocko_events.yaml"), format="yaml")

        stations = self.get_stations()
        stations.export_pyrocko_stations(path.parent / "stations.yaml")
        logger.info("exported Pyrocko models to %s", path)

    def export_traces(
        self,
        directory: Path,
        batch_length: timedelta = timedelta(minutes=10),
    ) -> None:
        """Export traces for all events in the catalog.

        Args:
            directory (Path): Directory to export the traces to.
            batch_length (timedelta): Length of each batch of traces.
        """
        directory.mkdir(parents=True, exist_ok=True)
        start_time, end_time = self.get_time_span()
        current_start = start_time

        logger.info("exporting synthetic traces to %s", directory)
        while current_start < end_time:
            current_end = min(current_start + batch_length, end_time)
            traces = self.get_all_traces(
                start_time=current_start,
                end_time=current_end,
            )
            if not traces:
                continue
            save(
                traces,
                str(directory / "traces_%(network)s.%(station)s.%(location)s.mseed"),
                append=True,
            )

            current_start = current_end

    def export_catalog(self, directory: Path) -> None:
        """Export the synthetic event catalog to CSV and Pyrocko events file.

        Args:
            directory (Path): Directory to export the catalog to.
        """
        logger.info("exporting synthetic event catalog to %s", directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.export_csv(directory)
        self.export_pyrocko(directory)
        self.export_traces(directory / "traces")

        catalog = directory / "catalog.json"
        catalog.write_text(self.model_dump_json(indent=2))

    def get_time_span(self) -> tuple[datetime, datetime]:
        """Get the time span of all events in the catalog.

        Returns:
            tuple[datetime, datetime]: Start and end time of the events.
        """
        if not self.events:
            raise ValueError("No events in the catalog.")
        last_arrivals = [event.last_arrival() for event in self.events]

        start_time = min(event.origin_time for event in self.events)
        end_time = max(arr for arr in last_arrivals if arr is not None)
        return start_time, end_time

    def get_stations(self) -> Stations:
        stations = {}
        for event in self.events:
            for receiver in event.get_receivers():
                stations[receiver.nsl] = receiver.as_station()
        return Stations.model_construct(stations=list(stations.values()))
