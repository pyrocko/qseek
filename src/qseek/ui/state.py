from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
from nicegui import Event, app, binding

from qseek.models.catalog import EventCatalog
from qseek.ui.explorer.base import RunSource
from qseek.ui.models import EventMinimal

logger = logging.getLogger(__name__)


class Filter:
    range: dict = binding.BindableProperty()
    data_min: float = binding.BindableProperty()
    data_max: float = binding.BindableProperty()
    user_defined: bool = False

    event_attribute: str = ""

    def __init__(self):
        self.range = {"min": 0.0, "max": 1.0}
        self.data_min = 0.0
        self.data_max = 1.0
        self.user_defined = False

    def filter(self, event: EventMinimal) -> bool:
        if not self.user_defined:
            return False
        val = getattr(event, self.event_attribute)
        return not (self.range["min"] <= val <= self.range["max"])

    def set_user_defined(self, user_defined: bool = True):
        self.user_defined = user_defined

    def reset(self, events: list[EventMinimal]) -> None:
        if not events:
            return
        values = np.array(
            [getattr(ev, self.event_attribute) for ev in events],
            dtype=float,
        )
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return
        self.data_min = float(values.min())
        self.data_max = float(values.max())
        self.range = {"min": self.data_min, "max": self.data_max}
        self.user_defined = False

    def set_live(self, live: bool = False): ...


class SemblanceFilter(Filter):
    event_attribute = "semblance"


class MagnitudeFilter(Filter):
    def filter(self, event: EventMinimal) -> bool:
        if not self.user_defined:
            return False
        if event.magnitude is None:
            return True
        return not (self.range["min"] <= event.magnitude.average <= self.range["max"])

    def reset(self, events: list[EventMinimal]) -> None:
        values = np.array(
            [ev.magnitude.average for ev in events if ev.magnitude is not None],
            dtype=float,
        )
        if len(values) == 0:
            self.data_min = 0.0
            self.data_max = 0.0
        else:
            self.data_min = float(values.min())
            self.data_max = float(values.max())
        self.range = {"min": self.data_min, "max": self.data_max}
        self.user_defined = False


class RMSFilter(Filter):
    event_attribute = "rms"


class DepthFilter(Filter):
    event_attribute = "depth"


class NPicksFilter(Filter):
    event_attribute = "n_picks"


class TimeFilter(Filter):
    def filter(self, event: EventMinimal) -> bool:
        if not self.user_defined:
            return False
        value = self.range
        if not isinstance(value, dict) or "from" not in value:
            return False
        min_dt = datetime.fromisoformat(value["from"]).replace(tzinfo=timezone.utc)
        # add one day to make the end date inclusive
        max_dt = datetime.fromisoformat(value["to"]).replace(
            tzinfo=timezone.utc
        ) + timedelta(days=1)
        return not (min_dt <= event.time <= max_dt)

    def reset(self, events: list[EventMinimal]) -> None:
        if not events:
            return
        times = [ev.time for ev in events]
        self.data_min = min(times).timestamp()
        self.data_max = max(times).timestamp()
        self.range = {
            "from": min(times).strftime("%Y-%m-%d"),
            "to": max(times).strftime("%Y-%m-%d"),
        }
        self.user_defined = False


class CatalogStore:
    MAX_VISIBLE_EVENTS: int = 10_000

    events: list[EventMinimal] = binding.BindableProperty()
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
    new_events: Event[list[EventMinimal]]

    _all_events: list[EventMinimal] = []
    _catalog: EventCatalog | None = None
    _run: RunSource | None = None

    def __init__(self):
        self.events = []

        self.semblance_filter = SemblanceFilter()
        self.magnitude_filter = MagnitudeFilter()
        self.rms_filter = RMSFilter()
        self.depth_filter = DepthFilter()
        self.n_picks_filter = NPicksFilter()
        self.time_filter = TimeFilter()

        self.filters: list[Filter] = [
            self.time_filter,
            self.semblance_filter,
            self.magnitude_filter,
            self.rms_filter,
            self.depth_filter,
            self.n_picks_filter,
        ]

        self._run_watcher: asyncio.Task | None = None
        self.updated = Event()
        self.new_events = Event()

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

    async def _watch_run(self, refresh_interval: float = 5.0):
        if self._run is None:
            raise RuntimeError("No run set for watching")
        logger.info("Starting run watcher for run %s", self._run.name)

        last_update = time.time()

        while True:
            async with self._run.updated:
                if len(self._catalog.events) == len(self._all_events):
                    await self._run.updated.wait()
            time_since_update = time.time() - last_update
            if time_since_update < refresh_interval:
                await asyncio.sleep(refresh_interval - time_since_update)
            last_update = time.time()

            new_events = [
                EventMinimal.from_event(ev)
                for ev in self._catalog.events[len(self._all_events) :]
            ]
            self._all_events += new_events
            self.reset_filters(reset_user_filters=False)
            filtered_events = self.filter_events(new_events)
            if not filtered_events:
                continue

            self.new_events.emit(filtered_events)

    def filter_events(
        self,
        new_events: list[EventMinimal] | None = None,
    ) -> list[EventMinimal]:
        if self._catalog is None:
            raise RuntimeError("No catalog set for filtering")

        events = new_events if new_events is not None else self._all_events
        filtered_events = [
            e for e in events if not any(f.filter(e) for f in self.filters)
        ]

        if new_events is None:
            # events are chronological ascending, keep the most recent tail
            visible_events = filtered_events[-self.MAX_VISIBLE_EVENTS :]
        else:
            # new list assignment to trigger BindableProperty
            visible_events = self.events + filtered_events
            overflow = len(visible_events) - self.MAX_VISIBLE_EVENTS
            if overflow > 0:
                visible_events = visible_events[overflow:]

        self.events = visible_events
        self.times = [ev.time for ev in visible_events]
        self.uids = [ev.uid for ev in visible_events]

        self.refresh_caches()
        return filtered_events

    def refresh_caches(self):
        self.magnitudes = np.array(
            [
                ev.magnitude.average if ev.magnitude is not None else np.nan
                for ev in self.events
            ],
            dtype=float,
        )
        if not self.events:
            self.lats = np.array([])
            self.lons = np.array([])
            self.depths = np.array([])
            self.north_shifts = np.array([])
            self.east_shifts = np.array([])
            self.semblances = np.array([])
            self.n_picks = np.array([])
            return

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

    def get_event_by_uid(self, uid: UUID) -> EventMinimal:
        for ev in self._all_events:
            if ev.uid == uid:
                return ev
        raise ValueError(f"Event with uid {uid} not found")

    def reset_filters(self, reset_user_filters: bool = False):
        if not reset_user_filters and any(f.user_defined for f in self.filters):
            return
        for f in self.filters:
            f.reset(self._all_events)

    def has_magnitudes(self) -> bool:
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
