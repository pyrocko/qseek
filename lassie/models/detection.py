from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from lassie.models.location import Location


class Detection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    detection_peak: float


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
                f"{lat:.4f}, {lon:.4f}, {det.effective_elevation:.4f},"
                f" {det.detection_peak}, {det.time}"
            )
        file.write_text("\n".join(lines))

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __iadd__(self, other: Detections) -> Detections:
        self.detections.extend(other.detections)
        return self
