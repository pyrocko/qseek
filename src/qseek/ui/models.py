import contextlib
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np
from nicegui import app
from pydantic import ValidationError

from qseek.models.catalog import EventCatalog
from qseek.models.detection import EventDetection
from qseek.search import Search

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventMinimal:
    uid: UUID
    lat: float
    lon: float
    depth: float
    time: datetime
    semblance: float
    n_picks: int
    magnitude: float
    event: EventDetection

    def as_tuple(self) -> tuple[float, float, float, datetime, float, float, float]:
        return (
            self.lat,
            self.lon,
            self.depth,
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
    magnitudes: np.ndarray
    semblances: np.ndarray

    def __init__(self, catalog: EventCatalog):
        self.catalog = catalog
        self.events = [
            EventMinimal(
                uid=ev.uid,
                lat=ev.effective_lat,
                lon=ev.effective_lon,
                depth=ev.depth,
                time=ev.time,
                semblance=ev.semblance,
                n_picks=ev.n_picks,
                magnitude=ev.magnitude or ev.semblance,
                event=ev,
            )
            for ev in catalog.events
        ]

        (
            self.lats,
            self.lons,
            self.depths,
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


class RunProxy:
    path: Path
    n_events: int
    hash: str

    _search: Search | None = None
    _catalog: CatalogProxy | None = None

    def __init__(self, path: Path) -> None:
        self.path = path
        search_file = path / "search.json"
        detections_file = path / "detections.json"
        if not search_file.is_file():
            raise ValueError(f"search.json not found in {path}")
        if not detections_file.is_file():
            raise ValueError(f"detections.json not found in {path}")

        self.hash = hashlib.sha1(search_file.read_bytes()).hexdigest()

    async def get_catalog(self) -> CatalogProxy:
        if not self._catalog:
            self._catalog = CatalogProxy(EventCatalog.load_rundir(self.path))
        return self._catalog

    def get_search(self) -> Search:
        if not self._search:
            search_file = self.path / "search.json"
            self._search = Search.model_validate_json(
                search_file.read_bytes(), context={"assume_validated": True}
            )
        return self._search


class RunManager:
    runs: dict[str, RunProxy]
    _callback: Callable[[], None] | None = None

    def __init__(self):
        self.runs = {}

    def add_dir(self, path: Path):
        if not path.is_dir():
            raise ValueError(f"Provided path {path} is not a directory")
        if not path.exists():
            raise ValueError(f"Provided path {path} does not exist")
        for search_json in path.glob("*/search.json"):
            run_path = search_json.parent
            try:
                run = RunProxy(run_path)
                self.runs[run.hash] = run

            except ValidationError as e:
                raise e

        logger.info("Loaded %d runs from %s", self.n_runs, path)

    def get_run(self, hash: str) -> RunProxy:
        return self.runs[hash]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def set_active_run(self, hash: str) -> None:
        if hash not in self.runs:
            raise ValueError(f"Run with hash {hash} not found")
        old_hash = app.storage.tab.get("active_run")
        app.storage.tab["active_run"] = hash
        if old_hash != hash and self._callback:
            logger.info("Active run changed to %s", hash)
            self._callback()

    def get_active_run(self) -> RunProxy:
        if self.n_runs == 0:
            raise RuntimeError("No runs loaded")
        try:
            active_run = app.storage.tab["active_run"]
        except (KeyError, RuntimeError):
            active_run = next(iter(self.runs.keys()))
            with contextlib.suppress(ValueError, RuntimeError):
                self.set_active_run(active_run)
        return self.get_run(active_run)

    def on_active_run_change(self, callback: Callable[[], None]) -> None:
        self._callback = callback
