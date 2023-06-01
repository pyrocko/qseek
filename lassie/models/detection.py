from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field
from pyrocko import io
from pyrocko.model import Event, dump_events

from lassie.models.location import Location
from lassie.models.station import Station
from lassie.octree import Octree
from lassie.plot.octree import plot_octree, plot_octree_surface

if TYPE_CHECKING:
    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)


class EventDetection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    semblance: float
    octree: Octree

    stations: list[Station] = []

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

    def add_semblance(self, trace: Trace) -> None:
        trace.set_station("SEMBL")
        io.save(
            trace,
            str(self.rundir / "semblance.mseed"),
            append=True,
        )

    def get_semblance(self, start_time: datetime, end_time: datetime) -> Trace:
        ...

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
        dump_events(
            [detection.as_pyrocko_event() for detection in self],
            filename=str(filename),
        )

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(self.detections)
