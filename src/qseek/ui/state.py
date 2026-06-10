from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import TypedDict
from uuid import UUID

import numpy as np
from nicegui import Event, app, binding

from qseek.models.catalog import EventCatalog
from qseek.ui.explorer.base import RunSource
from qseek.ui.models import EventMinimal

logger = logging.getLogger(__name__)


class GuiRange(TypedDict):
    min: float
    max: float


class CatalogStore:
    semblance_range: GuiRange = binding.BindableProperty()
    magnitude_range: GuiRange = binding.BindableProperty()
    depth_range: GuiRange = binding.BindableProperty()
    n_picks_range: GuiRange = binding.BindableProperty()
    date_range: dict = binding.BindableProperty()
    user_defined_filters: bool = False

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

    updated: Event
    _all_events: list[EventMinimal] = []
    _catalog: EventCatalog | None = None
    _run: RunSource | None = None

    def __init__(self):
        self.semblance_range = {
            "min": 0.0,
            "max": 2.0,
        }
        self.magnitude_range = {
            "min": -2.0,
            "max": 9.0,
        }
        self.depth_range = {
            "min": -10000.0,
            "max": 50_000.0,
        }
        self.n_picks_range = {
            "min": 0,
            "max": 100,
        }
        self.date_range = {
            "from": "1970-01-01",
            "to": "2050-01-01",
        }

        self._run_watcher: asyncio.Task | None = None
        self.updated = Event()

    @property
    def full_catalog(self) -> EventCatalog:
        if self._catalog is None:
            raise RuntimeError("No catalog loaded")
        return self._catalog

    async def attach(self, run: RunSource):
        await self.detach()
        self._catalog = await run.get_catalog()
        await run.attach(self)
        self._all_events = [EventMinimal.from_event(ev) for ev in self._catalog.events]
        logger.debug("Run %s loaded with %d events", run.name, self.n_events)

        self.reset_filters(reset_user_filters=True)
        self.filter_events()
        self.refresh_caches()

        self.updated.emit()

        if self._run_watcher is not None:
            self._run_watcher.cancel()
        self._run = run
        self._run_watcher = asyncio.create_task(self._watch_run())

    async def detach(self):
        if self._run is not None:
            await self._run.detach(self)
        if self._run_watcher is not None:
            self._run_watcher.cancel()
            self._run_watcher = None

        self._catalog = None
        self._all_events = []
        self.events = []
        self.uids = []
        self.times = []
        self.semblances = np.array([])
        self.magnitudes = np.array([])
        self.lats = np.array([])
        self.lons = np.array([])
        self.depths = np.array([])
        self.n_picks = np.array([])
        self.east_shifts = np.array([])
        self.north_shifts = np.array([])

    def has_catalog(self) -> bool:
        return self._catalog is not None

    async def _watch_run(self, refresh_interval: float = 10.0):
        if self._run is None:
            raise RuntimeError("No run set for watching")
        logger.info("Starting run watcher for run %s", self._run.name)
        last_update = time.time()
        while True:
            await self._run.updated.wait()
            self._all_events += [
                EventMinimal.from_event(ev)
                for ev in self._catalog.events[len(self._all_events) :]
            ]
            self.reset_filters(reset_user_filters=False)
            self.filter_events()
            self.refresh_caches()

            self.updated.emit()

            time_since_update = time.time() - last_update
            last_update = time.time()

            if time_since_update < refresh_interval:
                await asyncio.sleep(refresh_interval - time_since_update)

    def filter_events(self):
        if self._catalog is None:
            raise RuntimeError("No catalog set for filtering")

        date_min = datetime.strptime(self.date_range["from"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_max = datetime.strptime(self.date_range["to"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        self.events = [
            e
            for e in self._all_events
            if self.semblance_range["min"] <= e.semblance <= self.semblance_range["max"]
            and self.depth_range["min"] <= e.depth <= self.depth_range["max"]
            and self.n_picks_range["min"] <= e.n_picks <= self.n_picks_range["max"]
            and date_min <= e.time <= date_max + timedelta(days=1)
        ]

        if self.has_magnitudes():
            self.events = [
                e
                for e in self.events
                if self.magnitude_range["min"]
                <= (e.magnitude.average if e.magnitude is not None else np.nan)
                <= self.magnitude_range["max"]
            ]

        self.times = [ev.time for ev in self.events]
        self.uids = [ev.uid for ev in self.events]

    def refresh_caches(self):
        self.magnitudes = np.array(
            [
                ev.magnitude.average if ev.magnitude is not None else np.nan
                for ev in self.events
            ],
            dtype=float,
        )
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

        self.updated.emit()

    def get_event_by_uid(self, uid: UUID) -> EventMinimal:
        for ev in self.events:
            if ev.uid == uid:
                return ev
        raise ValueError(f"Event with uid {uid} not found")

    def reset_filters(self, reset_user_filters: bool = False):
        if not reset_user_filters and self.user_defined_filters:
            return

        if reset_user_filters:
            self.user_defined_filters = False

        semblances = np.array([ev.semblance for ev in self._all_events])
        magnitudes = np.array(
            [
                ev.magnitude.average if ev.magnitude is not None else np.nan
                for ev in self._all_events
            ]
        )
        depths = np.array([ev.depth for ev in self._all_events])
        n_picks = np.array([ev.n_picks for ev in self._all_events])
        times = [ev.time for ev in self._all_events]

        self.semblance_range = {
            "min": float(semblances.min()),
            "max": float(semblances.max()),
        }
        self.magnitude_range = {
            "min": float(np.nanmin(magnitudes)),
            "max": float(np.nanmax(magnitudes)),
        }
        self.depth_range = {
            "min": float(depths.min()),
            "max": float(depths.max()),
        }
        self.n_picks_range = {
            "min": int(n_picks.min()),
            "max": int(n_picks.max()),
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
    run_id: str = binding.BindableProperty()
    loading: str = binding.BindableProperty()

    catalog_store: CatalogStore

    def __init__(self, run: RunSource):
        self._init_lock = asyncio.Lock()
        self.run = run
        self.run_name = run.name
        self.run_id = run.hash
        self.loading = ""

        self.catalog_store = CatalogStore()

        self.run_changed = Event()

    async def set_run(self, run: RunSource):
        self.run = run
        self.run_name = run.name
        self.run_id = run.hash
        self.run_changed.emit()
        self.loading = ""

    @contextmanager
    def loading_message(self, message: str):
        self.loading = message
        yield
        self.loading = ""

    async def get_catalog(self) -> CatalogStore:
        async with self._init_lock:
            if self.run is None:
                raise RuntimeError("No run set for filtering")
            if not self.catalog_store.has_catalog():
                with self.loading_message(f"Loading run {self.run.name}..."):
                    await self.catalog_store.attach(self.run)
        return self.catalog_store

    async def clear(self):
        await self.catalog_store.detach()


def get_tab_state() -> TabState:
    """Get the state for the current tab."""
    if "state" not in app.storage.tab:
        raise RuntimeError("Tab state does not exist")
    return app.storage.tab["state"]


def create_tab_state(run: RunSource) -> TabState:
    """Create a new tab state with the given default run."""
    if "state" not in app.storage.tab:
        state = TabState(run)
        app.storage.tab["state"] = state
    return app.storage.tab["state"]
