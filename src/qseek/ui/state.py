from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np
from nicegui import Event, app, binding

from qseek.models.catalog import EventCatalog
from qseek.ui.explorer.base import RunSource
from qseek.ui.models import EventMinimal


class FilteredCatalog:
    semblance_min: float = binding.BindableProperty()
    semblance_max: float = binding.BindableProperty()

    magnitudes_min: float = binding.BindableProperty()
    magnitudes_max: float = binding.BindableProperty()

    date_min: datetime = binding.BindableProperty()
    date_max: datetime = binding.BindableProperty()

    events: list[EventMinimal] = []
    uids: list[UUID] = []
    times: list[datetime] = []
    semblances: np.ndarray = np.array([])
    magnitudes: np.ndarray = np.array([])
    lats: np.ndarray = np.array([])
    lons: np.ndarray = np.array([])
    depths: np.ndarray = np.array([])
    east_shifts: np.ndarray = np.array([])
    north_shifts: np.ndarray = np.array([])

    _catalog: EventCatalog | None = None

    def __init__(self):
        self.semblance_min = 0.0
        self.semblance_max = 2.0
        self.magnitudes_min = -1.0
        self.magnitudes_max = 9.0

        self.date_min = datetime.fromisoformat("1970-01-01T00:00:00Z")
        self.date_max = datetime.fromisoformat("2050-01-01T00:00:00Z")

        self.updated = Event()

    async def set_run(self, run: RunSource):
        self._catalog = await run.get_catalog()
        self._update()

    def _update(self):
        if self._catalog is None:
            raise RuntimeError("No catalog set for filtering")

        self.events = [
            EventMinimal.from_event(ev)
            for ev in self._catalog.events
            if self.semblance_min <= ev.semblance <= self.semblance_max
            and self.date_min <= ev.time <= self.date_max
        ]

        self.times = [ev.time for ev in self.events]
        self.uids = [ev.uid for ev in self.events]

        mags = [
            ev.magnitude.average if ev.magnitude is not None else np.nan
            for ev in self.events
        ]
        self.magnitudes = np.array(mags, dtype=float)

        (
            self.lats,
            self.lons,
            self.depths,
            self.north_shifts,
            self.east_shifts,
            _,
            self.semblances,
            self.n_picks,
            *_,
        ) = map(np.array, zip(*(ev.as_tuple() for ev in self.events), strict=True))

        self.updated.emit()

    def get_event_by_uid(self, uid: UUID) -> EventMinimal:
        for ev in self.events:
            if ev.uid == uid:
                return ev
        raise ValueError(f"Event with uid {uid} not found")

    def has_magnitudes(self):
        return not np.all(np.isnan(self.magnitudes))

    @property
    def n_events(self) -> int:
        return len(self.events)


class TabState:
    run: RunSource
    run_name: str = binding.BindableProperty()

    filtered_catalog: FilteredCatalog

    _default_run: RunSource | None = None

    def __init__(self):
        if self._default_run is None:
            raise RuntimeError("No default run set")

        self.run = self._default_run
        self.run_name = self.run.name
        self.filtered_catalog = FilteredCatalog()

        self.run_changed = Event()
        self.catalog_updated = Event()

    async def set_run(self, run: RunSource):
        self.run = run
        self.run_name = run.name
        await self.filtered_catalog.set_run(run)
        self.run_changed.emit()

    async def get_filtered_catalog(self) -> FilteredCatalog:
        if self.filtered_catalog._catalog is None:
            await self.filtered_catalog.set_run(self.run)
        return self.filtered_catalog

    @classmethod
    def set_default_run(cls, run: RunSource):
        cls._default_run = run


def get_tab_state() -> TabState:
    if "state" not in app.storage.tab:
        app.storage.tab["state"] = TabState()
    return app.storage.tab["state"]
