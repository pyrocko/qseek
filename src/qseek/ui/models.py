import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import numpy as np

from qseek.magnitudes.base import EventMagnitude
from qseek.models.catalog import EventCatalog
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
    magnitude: float | None
    event: EventDetection

    def as_tuple(
        self,
    ) -> tuple[float, float, float, float, float, datetime, float, float, float]:
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


class CatalogProxy:
    catalog: EventCatalog

    events: list[EventMinimal]
    uids: list[UUID]
    lats: np.ndarray
    lons: np.ndarray
    depths: np.ndarray
    times: list[datetime]
    magnitudes: EventMagnitude | None
    semblances: np.ndarray

    def __init__(self, catalog: EventCatalog):
        self.catalog = catalog
        self.events = [
            EventMinimal(
                uid=ev.uid,
                lat=ev.effective_lat,
                lon=ev.effective_lon,
                north_shift=ev.north_shift,
                east_shift=ev.east_shift,
                depth=ev.depth,
                time=ev.time,
                semblance=ev.semblance,
                n_picks=ev.n_picks,
                magnitude=ev.magnitude,
                event=ev,
            )
            for ev in catalog.events
        ]

        (
            self.lats,
            self.lons,
            self.depths,
            self.north_shift,
            self.east_shift,
            _,
            self.semblances,
            self.n_picks,
            self.magnitudes,
        ) = map(np.array, zip(*(ev.as_tuple() for ev in self.events), strict=True))
        self.times = [ev.time for ev in self.events]
        self.uids = [ev.uid for ev in self.events]

    def get_event_by_uid(self, uid: UUID) -> EventMinimal:
        for ev in self.events:
            if ev.uid == uid:
                return ev
        raise ValueError(f"Event with uid {uid} not found")

    @property
    def n_events(self) -> int:
        return len(self.events)
