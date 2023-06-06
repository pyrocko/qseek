from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Self
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import Event, dump_events

from lassie.features_receiver.base import ReceiverFeature
from lassie.images import ImageFunctionPick
from lassie.models.location import Location
from lassie.models.station import Station
from lassie.octree import Octree
from lassie.plot.octree import plot_octree, plot_octree_surface
from lassie.tracers import RayTracerArrival
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.images.base import PickedArrival, WaveformImage

logger = logging.getLogger(__name__)


class PhaseReceiver(Station):
    arrival_model: RayTracerArrival
    arrival_observed: ImageFunctionPick | None = None
    features: list[ReceiverFeature] = []

    @property
    def traveltime_delay(self) -> timedelta | None:
        if self.arrival_observed:
            return self.arrival_model.time - self.arrival_observed.time

    @classmethod
    def from_station(cls, station: Station, arrival_model: RayTracerArrival) -> Self:
        return cls(arrival_model=arrival_model, **station.dict())

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        pick_markers = [
            marker.PhaseMarker(
                nslc_ids=[(*self.nsl, "*")],
                tmax=self.arrival_model.time.timestamp(),
                tmin=self.arrival_model.time.timestamp(),
                kind=0,
                phasename="mod",
            )
        ]
        if self.arrival_observed:
            pick_markers.append(
                marker.PhaseMarker(
                    nslc_ids=[(*self.nsl, "*")],
                    tmax=self.arrival_observed.time.timestamp(),
                    tmin=self.arrival_observed.time.timestamp(),
                    automatic=True,
                    kind=1,
                    phasename="obs",
                )
            )
        return pick_markers

    def get_waveforms(
        self,
        squirrel: Squirrel,
        seconds_after: float = 5.0,
        seconds_before: float = 0.0,
    ) -> list[Trace]:
        arrival = self.arrival_observed or self.arrival_model
        traces = squirrel.get_waveforms(
            codes=[(*self.nsl, "*")],
            tmin=(arrival.time - timedelta(seconds=seconds_before)).timestamp(),
            tmax=(arrival.time + timedelta(seconds=seconds_after)).timestamp(),
            want_incomplete=False,
        )
        if not traces:
            raise KeyError
        return traces

    def add_feature(self, feature: ReceiverFeature) -> None:
        self.features.append(feature)


class PhaseDetection(BaseModel):
    phase: PhaseDescription
    receivers: list[PhaseReceiver] = []

    @classmethod
    def from_image(
        cls,
        image: WaveformImage,
        arrivals_model: list[RayTracerArrival],
        arrivals_observed: list[PickedArrival | None] | None = None,
    ) -> Self:
        if image.stations.n_stations != len(arrivals_model):
            raise ValueError("Number of receivers and traveltimes missmatch.")

        detections = cls(
            phase=image.phase,
            receivers=[
                PhaseReceiver.from_station(sta, traveltime)
                for sta, traveltime in zip(image.stations, arrivals_model)
            ],
        )
        if arrivals_observed is not None:
            detections.set_arrivals_observed(arrivals_observed)
        return detections

    def set_arrivals_observed(self, arrivals: list[PickedArrival | None]) -> None:
        for receiver, arrival in zip(self.receivers, arrivals, strict=True):
            receiver.arrival_observed = arrival

    def add_features(self, features: list[ReceiverFeature | None]) -> None:
        for receiver, feature in zip(self.receivers, features, strict=True):
            if feature:
                receiver.add_feature(feature)

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        pick_markers = []
        for receiver in self:
            for pick in receiver.as_pyrocko_markers():
                pick.set_phasename(f"{self.phase[-1]}{pick.get_phasename()}")
                pick_markers.append(pick)
        return pick_markers

    def save_csv(self, filename: Path) -> None:
        with filename.open("w") as file:
            file.write("lat, lon, elevation, traveltime_delay")
            for receiver in self:
                file.write(
                    f"{receiver.effective_lat}, {receiver.effective_lon}, {receiver.effective_elevation}, {receiver.traveltime_delay or 'nan'}"  # noqa
                )

    def __iter__(self) -> Iterator[PhaseReceiver]:
        return iter(self.receivers)


class EventDetection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    semblance: float
    octree: Octree

    arrivals: list[PhaseDetection]

    def as_pyrocko_event(self) -> Event:
        return Event(
            name=self.time.isoformat(sep="T"),
            time=self.time.timestamp(),
            lat=self.lat,
            lon=self.lon,
            east_shift=self.east_shift,
            north_shift=self.north_shift,
            depth=self.depth,
            elevation=self.elevation,
            magnitude=self.semblance,
            magnitude_type="semblance",
        )

    def as_pyrocko_markers(self) -> list[marker.EventMarker | marker.PhaseMarker]:
        event = self.as_pyrocko_event()

        pyrocko_markers: list[marker.EventMarker | marker.PhaseMarker] = [
            marker.EventMarker(event)
        ]
        for detection in self.arrivals:
            for pick in detection.as_pyrocko_markers():
                pick.set_event(event)
                pyrocko_markers.append(pick)
        return pyrocko_markers

    def save_pyrocko_markers(self, filename: Path) -> None:
        logger.info("saving detection's Pyrocko markers to %s", filename)
        marker.save_markers(self.as_pyrocko_markers(), str(filename))

    def get_detection(self, phase: PhaseDescription) -> PhaseDetection:
        """Get detecion ending with phase name.

        Args:
            phase (str): Detection phase name, e.g. 'cake:P'.

        Raises:
            ValueError: Raised when the phase cannot be found.

        Returns:
            PhaseDetection: The requested phase detection.
        """
        for detection in self.arrivals:
            if detection.phase == phase:
                return detection
        raise ValueError(
            f"No phase ending with {phase} found. "
            f"Select from {', '.join(det.phase for det in self.arrivals)}."
        )

    def plot(self, cmap: str = "Oranges") -> None:
        plot_octree(self.octree, cmap=cmap)

    def plot_surface(
        self, accumulator: Callable = np.max, cmap: str = "Oranges"
    ) -> None:
        plot_octree_surface(self.octree, accumulator=accumulator, cmap=cmap)


class Detections(BaseModel):
    rundir: Path
    detections: list[EventDetection] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.detections_dir.exists():
            self.detections_dir.mkdir()
            logger.info("created directory %s", self.detections_dir)
        else:
            logger.info("loading detections from %s", self.detections_dir)
            self.load_detections()

    @property
    def detections_dir(self) -> Path:
        return self.rundir / "detections"

    def add(self, detection: EventDetection) -> None:
        detection.octree.make_concrete()
        self.detections.append(detection)

        filename = self.detections_dir / (str(detection.uid) + ".json")
        filename.write_text(detection.json())

        self.save_csv(self.rundir / "detections.csv")
        self.save_pyrocko_events(self.rundir / "pyrocko-events.list")
        self.save_pyrocko_markers(self.rundir / "pyrocko-markers.list")

    def add_semblance(self, trace: Trace) -> None:
        trace.set_station("SEMBL")
        io.save(
            trace,
            str(self.rundir / "semblance.mseed"),
            append=True,
        )

    def load_detections(self) -> None:
        for file in self.detections_dir.glob("*.json"):
            detection = EventDetection.parse_file(file)
            self.detections.append(detection)

    def get(self, uid: UUID) -> EventDetection:
        for detection in self:
            if detection.uid == uid:
                return detection
        raise KeyError("detection not found")

    def n_detections(self) -> int:
        return len(self.detections)

    def save_csv(self, file: Path) -> None:
        lines = ["lat, lon, depth, detection_peak, time"]
        for det in self:
            lat, lon = det.effective_lat_lon
            lines.append(
                f"{lat:.5f}, {lon:.5f}, {-det.effective_elevation:.1f},"
                f" {det.semblance}, {det.time}"
            )
        file.write_text("\n".join(lines))

    def save_pyrocko_events(self, filename: Path) -> None:
        logger.info("saving Pyrocko events to %s", filename)
        dump_events(
            [detection.as_pyrocko_event() for detection in self],
            filename=str(filename),
        )

    def save_pyrocko_markers(self, filename: Path) -> None:
        logger.info("saving Pyrocko markers to %s", filename)
        pyrocko_markers = []
        for detection in self:
            pyrocko_markers.extend(detection.as_pyrocko_markers())
        marker.save_markers(pyrocko_markers, str(filename))

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(self.detections)
