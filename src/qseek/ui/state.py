from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

import numpy as np
from nicegui import Event, app, binding

from qseek.models.catalog import EventCatalog
from qseek.ui.explorer.base import RunSource
from qseek.ui.models import EventMinimal


class NiceGuiRange(TypedDict):
    min: float
    max: float


class FilteredCatalog:
    semblance_range: NiceGuiRange = binding.BindableProperty()
    magnitude_range: NiceGuiRange = binding.BindableProperty()
    depth_range: NiceGuiRange = binding.BindableProperty()
    n_picks_range: NiceGuiRange = binding.BindableProperty()
    date_range: dict = binding.BindableProperty()

    events: list[EventMinimal] = []
    uids: list[UUID] = []
    times: list[datetime] = []
    semblances: np.ndarray = np.array([])
    magnitudes: np.ndarray = np.array([])
    lats: np.ndarray = np.array([])
    lons: np.ndarray = np.array([])
    depths: np.ndarray = np.array([])
    n_picks: np.ndarray = np.array([])
    east_shifts: np.ndarray = np.array([])
    north_shifts: np.ndarray = np.array([])

    _all_events: list[EventMinimal] = []
    _catalog: EventCatalog | None = None

    def __init__(self):
        self.semblance_range = {
            "min": 0.0,
            "max": 2.0,
        }
        self.magnitude_range = {
            "min": -1.0,
            "max": 9.0,
        }
        self.depth_range = {
            "min": 0.0,
            "max": 50_000.0,
        }
        self.n_picks_range = {
            "min": 0.0,
            "max": 100.0,
        }
        self.date_range = {
            "from": "1970-01-01",
            "to": "2050-01-01",
        }

        self.updated = Event()

    async def set_run(self, run: RunSource):
        self._catalog = await run.get_catalog()
        self._all_events = [EventMinimal.from_event(ev) for ev in self._catalog.events]
        self.reset_filters()
        self._refresh_event_data()

    def _refresh_event_data(self):
        if self._catalog is None:
            raise RuntimeError("No catalog set for filtering")

        date_min = datetime.strptime(self.date_range["from"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_max = datetime.strptime(self.date_range["to"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        self.events = [
            ev
            for ev in self._all_events
            if self.semblance_range["min"]
            <= ev.semblance
            <= self.semblance_range["max"]
            and self.depth_range["min"] <= ev.depth <= self.depth_range["max"]
            and self.n_picks_range["min"] <= ev.n_picks <= self.n_picks_range["max"]
            and date_min <= ev.time <= date_max
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

        self.lats = self.lats.astype(np.float32)
        self.lons = self.lons.astype(np.float32)
        self.depths = self.depths.astype(np.float32)
        self.north_shifts = self.north_shifts.astype(np.float32)
        self.east_shifts = self.east_shifts.astype(np.float32)
        self.semblances = self.semblances.astype(np.float32)
        self.n_picks = self.n_picks.astype(int)

        self.reset_filters()
        self.updated.emit()

    def get_event_by_uid(self, uid: UUID) -> EventMinimal:
        for ev in self.events:
            if ev.uid == uid:
                return ev
        raise ValueError(f"Event with uid {uid} not found")

    def reset_filters(self):
        semblances = np.array([ev.semblance for ev in self._all_events])
        depths = np.array([ev.depth for ev in self._all_events])
        n_picks = np.array([ev.n_picks for ev in self._all_events])
        times = [ev.time for ev in self._all_events]

        self.semblance_range = {
            "min": semblances.min(),
            "max": semblances.max(),
        }
        self.magnitude_range = {
            "min": -1,
            "max": 7,
        }
        self.depth_range = {
            "min": depths.min(),
            "max": depths.max(),
        }
        self.n_picks_range = {
            "min": n_picks.min(),
            "max": n_picks.max(),
        }

        self.date_range = {
            "from": min(times).strftime("%Y-%m-%d"),
            "to": max(times).strftime("%Y-%m-%d"),
        }

    def has_magnitudes(self):
        return not np.all(np.isnan(self.magnitudes))

    @property
    def n_events(self) -> int:
        return len(self.events)


class TabState:
    run: RunSource
    run_name: str = binding.BindableProperty()
    loading: bool = binding.BindableProperty()

    filtered_catalog: FilteredCatalog

    _default_run: RunSource | None = None

    def __init__(self):
        if self._default_run is None:
            raise RuntimeError("No default run set")

        self.loading = True

        self.run = self._default_run
        self.run_name = self._default_run.name

        self.filtered_catalog = FilteredCatalog()

        self.run_changed = Event()
        self.catalog_updated = Event()

    async def set_run(self, run: RunSource):
        self.run = run
        self.run_name = run.name
        self.loading = True
        await self.filtered_catalog.set_run(run)
        self.run_changed.emit()
        self.loading = False

    async def get_filtered_catalog(self) -> FilteredCatalog:
        if self.filtered_catalog._catalog is None:
            self.loading = True
            await self.filtered_catalog.set_run(self.run)
            self.loading = False
        return self.filtered_catalog

    @classmethod
    def set_default_run(cls, run: RunSource):
        cls._default_run = run


def get_tab_state() -> TabState:
    if "state" not in app.storage.tab:
        app.storage.tab["state"] = TabState()
    return app.storage.tab["state"]
