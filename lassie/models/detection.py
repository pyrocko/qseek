from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field

from lassie.models.location import Location
from lassie.octree import Octree


class Detection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    octree: Octree
    detection_peak: float

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
    detections: list[Detection] = []

    def add_detection(self, detection: Detection) -> None:
        self.detections.append(detection)

    def append(self, other: Detections) -> Detections:
        self.detections.extend(other.detections)
        return self

    def n_detections(self) -> int:
        return len(self.detections)

    def to_csv(self, file: Path) -> None:
        lines = ["lat, lon, elevation, detection_peak, time"]
        for det in self:
            lat, lon = det.effective_lat_lon
            lines.append(
                f"{lat:.5f}, {lon:.5f}, {det.effective_elevation:.1f},"
                f" {det.detection_peak}, {det.time}"
            )
        file.write_text("\n".join(lines))

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __iadd__(self, other: Detections) -> Detections:
        self.detections.extend(other.detections)
        return self
