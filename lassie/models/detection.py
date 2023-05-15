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

if TYPE_CHECKING:
    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)


class Detection(Location):
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
        )

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = self.octree.get_coordinates("cartesian").T
        ax.scatter(
            coords[0],
            coords[1],
            coords[2],
            c=self.octree.semblance,
            cmap="Oranges",
        )
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.set_zlabel("depth [m]")
        plt.show()

    def plot_surface(self, accumulator: Callable = np.max) -> None:
        import matplotlib.pyplot as plt

        surface = self.octree.reduce_surface(accumulator)
        ax = plt.figure().gca()
        ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 2], cmap="Oranges")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        plt.show()


class Detections(BaseModel):
    rundir: Path

    detections: list[Detection] = []

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

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)
        filename = self.detections_dir / (str(detection.uid) + ".json")
        filename.write_text(detection.json())
        logger.info("new detection at %s", detection.time)

        self.save_csv(self.rundir / "detections.csv")
        self.save_pyrocko_events(self.rundir / "events.list")

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
            detection = Detection.parse_file(file)
            self.detections.append(detection)

    def get(self, uid: UUID) -> Detection:
        for detection in self:
            if detection.uid == uid:
                return detection
        raise KeyError("detection not found")

    def n_detections(self) -> int:
        return len(self.detections)

    def save_csv(self, file: Path) -> None:
        lines = ["lat, lon, elevation, detection_peak, time"]
        for det in self:
            lat, lon = det.effective_lat_lon
            lines.append(
                f"{lat:.5f}, {lon:.5f}, {det.effective_elevation:.1f},"
                f" {det.semblance}, {det.time}"
            )
        file.write_text("\n".join(lines))

    def save_pyrocko_events(self, filename: Path) -> None:
        dump_events(
            [detection.as_pyrocko_event() for detection in self],
            filename=str(filename),
        )

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)
