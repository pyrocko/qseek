import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import UUID

from qseek.magnitudes.base import EventStationMagnitude
from qseek.models.detection import EventDetection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventMinimal:
    uid: UUID
    lat: float
    lon: float
    north_shift: float
    east_shift: float
    depth: float
    time: datetime
    semblance: float
    n_picks: int
    magnitude: EventStationMagnitude | None
    event: EventDetection

    def as_tuple(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        datetime,
        float,
        float,
        EventStationMagnitude | None,
    ]:
        return (
            self.lat,
            self.lon,
            self.depth,
            self.north_shift,
            self.east_shift,
            self.time,
            self.semblance,
            self.n_picks,
            self.magnitude,
        )

    @classmethod
    def from_event(cls, event: EventDetection) -> Self:
        return cls(
            uid=event.uid,
            lat=event.effective_lat,
            lon=event.effective_lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            time=event.time,
            semblance=event.semblance,
            n_picks=event.n_picks,
            magnitude=event.magnitude,
            event=event,
        )
