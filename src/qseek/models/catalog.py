from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterator
from uuid import UUID

import aiofiles
import numpy as np
from pydantic import BaseModel, PrivateAttr, ValidationError, computed_field
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import Event, dump_events
from pyrocko.trace import Trace
from rich.progress import track
from rich.table import Table

from qseek.console import console
from qseek.models.detection import (
    FILENAME_DETECTIONS,
    FILENAME_RECEIVERS,
    EventDetection,
    logger,
)
from qseek.stats import Stats
from qseek.utils import Symbols, time_to_path

if TYPE_CHECKING:
    from pyrocko.gui.marker import EventMarker, PhaseMarker

UPDATE_LOCK = asyncio.Lock()


class ReceiverCache:
    file: Path
    lines: list[str] = []
    mtime: float | None = None

    def __init__(self, file: Path) -> None:
        self.file = file
        self.load()

    def load(self) -> None:
        if not self.file.exists():
            logger.debug("receiver cache %s does not exist", self.file)
            return
        logger.debug("loading receiver cache from %s", self.file)
        self.lines = self.file.read_text().splitlines()
        self.mtime = self.file.stat().st_mtime

    def _check_mtime(self) -> None:
        if self.mtime is None or self.mtime != self.file.stat().st_mtime:
            self.load()

    def get_line(self, row_index: int) -> str:
        """Retrieves the line at the specified row index.

        Args:
            row_index (int): The index of the row to retrieve.

        Returns:
            str: The line at the specified row index.
        """
        self._check_mtime()
        return self.lines[row_index]

    def find_uid(self, uid: UUID, start_idx: int = 0) -> tuple[int, str]:
        """Find the given UID in the lines and return its index and value.

        get_line should be prefered over this method.

        Args:
            uid (UUID): The UID to search for.
            start_idx (int, optional): The detection index to start the search from.
                Defaults to 0.

        Raises:
            KeyError: If the UID is not found in the lines.

        Returns:
            tuple[int, str]: A tuple containing the index and value of the found UID.
        """
        self._check_mtime()
        find_uid = str(uid)

        offset = 0
        n_lines = len(self.lines)

        # Search in both directions
        while True:
            left_idx = start_idx - offset
            right_idx = start_idx + offset
            if start_idx + offset >= n_lines and start_idx - offset < 0:
                break
            with suppress(IndexError):
                if left_idx >= 0 and find_uid in self.lines[left_idx]:
                    return left_idx, self.lines[left_idx]
            with suppress(IndexError):
                if right_idx < n_lines and find_uid in self.lines[right_idx]:
                    return right_idx, self.lines[right_idx]
            offset += 1

        raise KeyError(f"UID {find_uid} not found in receiver cache.")


class EventCatalogStats(Stats):
    n_detections: int = 0
    max_semblance: float = 0.0

    _position: int = PrivateAttr(10)
    _catalog: EventCatalog | None = PrivateAttr(None)

    def set_catalog(self, catalog: EventCatalog) -> None:
        self._catalog = catalog
        self.n_detections = catalog.n_events

    @property
    def magnitudes(self) -> list[float]:
        if not self._catalog:
            return []
        return [det.magnitude.average for det in self._catalog if det.magnitude]

    @computed_field
    def mean_semblance(self) -> float:
        if not self.n_detections or not self._catalog:
            return 0.0
        return (
            sum(detection.semblance for detection in self._catalog) / self.n_detections
        )

    @computed_field
    def magnitude_min(self) -> float:
        return min(self.magnitudes) if self.magnitudes else 0.0

    @computed_field
    def magnitude_max(self) -> float:
        return max(self.magnitudes) if self.magnitudes else 0.0

    def new_detection(self, detection: EventDetection) -> None:
        self.n_detections += 1
        self.max_semblance = max(self.max_semblance, detection.semblance)

    def _populate_table(self, table: Table) -> None:
        table.add_row("No. Detections", f"[bold]{self.n_detections} :fire:")
        table.add_row("Maximum semblance", f"{self.max_semblance:.4f}")


class EventCatalog(BaseModel):
    rundir: Path
    events: list[EventDetection] = []

    _stats: ClassVar[EventCatalogStats] = EventCatalogStats()
    _receiver_cache: ReceiverCache = PrivateAttr()
    _csv_header: list[str] = PrivateAttr([])

    def model_post_init(self, context: Any) -> None:
        self._receiver_cache = ReceiverCache(self.rundir / FILENAME_RECEIVERS)

    @property
    def n_events(self) -> int:
        """Number of detections."""
        return len(self.events)

    @property
    def markers_dir(self) -> Path:
        dir = self.rundir / "pyrocko_markers"
        dir.mkdir(exist_ok=True)
        return dir

    @property
    def csv_dir(self) -> Path:
        dir = self.rundir / "csv"
        dir.mkdir(exist_ok=True)
        return dir

    @property
    def detections_file(self) -> Path:
        return (self.rundir / FILENAME_DETECTIONS).absolute()

    @property
    def receivers_file(self) -> Path:
        return (self.rundir / FILENAME_RECEIVERS).absolute()

    def set_csv_header(self, header: list[str]) -> None:
        self._csv_header = header

    def get_stats(self) -> EventCatalogStats:
        if not self._stats:
            self._stats.set_catalog(self)
        return self._stats

    def sort(self) -> None:
        """Sort the detections by time."""
        self.events = sorted(self.events, key=lambda d: d.time)

    def get_event(self, uid: UUID):
        """Get an event by its UUID.

        Args:
            uid (UUID): The event UUID.

        Returns:
            EventDetection: The event detection object.
        """
        for detection in self.events:
            if detection.uid == uid:
                return detection
        raise KeyError(f"event with uid {uid} not found")

    async def filter_events_by_time(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        """Filter the detections based on the given time range.

        Args:
            start_time (datetime | None): Start time of the time range.
            end_time (datetime | None): End time of the time range.
        """
        if not self.events:
            return

        events = []
        if start_time is not None and min(det.time for det in self.events) < start_time:
            logger.info("filtering detections after start time %s", start_time)
            events = [det for det in self.events if det.time >= start_time]
        if end_time is not None and max(det.time for det in self.events) > end_time:
            logger.info("filtering detections before end time %s", end_time)
            events = [det for det in self.events if det.time <= end_time]
        if events:
            self.events = events
            self.get_stats().n_detections = len(self.events)
            await self.save()

    async def add(
        self,
        detection: EventDetection,
        jitter_location: float = 0.0,
    ) -> None:
        """Add a detection to the catalog.

        Args:
            detection (EventDetection): The detection to add.
            jitter_location (float, optional): Randomize the location of the detection
                by this many meters. This is only exported to the CSV
                and Pyrocko detections. Defaults to 0.0.
        """
        detection.set_index(self.n_events)
        detection.set_receiver_cache(self._receiver_cache)

        markers_file = self.markers_dir / f"{time_to_path(detection.time)}.list"
        self.markers_dir.mkdir(exist_ok=True)
        detection.export_pyrocko_markers(markers_file)

        self.events.append(detection)
        logger.info(
            "%s event detection %d %s at %.4f°, %.4f°, %.0f m, "
            "semblance %.3f, magnitude %.2f",
            Symbols.Target,
            self.n_events,
            str(detection.time)[:22],
            *detection.effective_lat_lon,
            detection.depth,
            detection.semblance,
            detection.magnitude.average if detection.magnitude else 0.0,
        )
        self.get_stats().new_detection(detection)
        # This has to happen after the markers are saved, cache is cleared
        await self.save_detection(detection, jitter_location=jitter_location)

    async def save_semblance_trace(self, trace: Trace) -> None:
        """Add semblance trace to detection and save to file.

        Data is multiplied by 1e3 and saved as int32 to leverage
        MiniSEED STEIM compression.

        Args:
            trace (Trace): semblance trace.
        """
        trace.set_station("SEMBL")
        trace.set_ydata((trace.ydata * 1e3).astype(np.int32))
        await asyncio.to_thread(
            io.save,
            trace,
            str(self.rundir / "semblance.mseed"),
            append=True,
        )

    @classmethod
    def last_modification(cls, rundir: Path) -> datetime:
        """Last modification of the event file.

        Returns:
            datetime: Last modification of the event file.
        """
        detection_file = rundir / FILENAME_DETECTIONS
        return datetime.fromtimestamp(
            detection_file.stat().st_mtime,
            tz=timezone.utc,
        )

    @classmethod
    def load_rundir(cls, rundir: Path) -> EventCatalog:
        """Load detections from files in the detections directory."""
        detection_file = rundir / FILENAME_DETECTIONS
        catalog = cls(rundir=rundir)

        if not detection_file.exists():
            logger.warning("cannot find %s", detection_file)
            return catalog

        with (
            console.status(f"loading detections from {rundir}..."),
            open(detection_file) as f,
        ):
            for idx, line in enumerate(f):
                try:
                    detection = EventDetection.model_validate_json(line)
                    detection.set_index(idx)
                    detection.set_receiver_cache(catalog._receiver_cache)
                    catalog.events.append(detection)
                except ValidationError as e:
                    logger.error("error loading detection %d: %s", idx, e)
                    logger.error("line: %s", line)

        logger.info("loaded %d detections", catalog.n_events)
        return catalog

    async def check(self, repair: bool = True) -> None:
        """Check the catalog for errors and inconsistencies.

        Args:
            repair (bool, optional): If True, attempt to repair the catalog.
                Defaults to True.
        """
        logger.info("checking catalog...")
        found_bad = 0
        found_duplicates = 0
        event_uids = set()

        def remove_event(detection: EventDetection) -> None:
            logger.warning("removing event %s", detection)
            self.events.remove(detection)

        for detection in track(
            self.events.copy(),
            description=f"checking {self.n_events} events...",
        ):
            try:
                _ = detection.receivers
            except ValueError:
                found_bad += 1
                if repair:
                    remove_event(detection)

            if detection.uid in event_uids:
                found_duplicates += 1
                if repair:
                    remove_event(detection)

            event_uids.add(detection.uid)

        if found_bad or found_duplicates:
            logger.info("found %d detections with invalid receivers", found_bad)
            logger.info("found %d duplicate detections", found_duplicates)
            if repair:
                logger.info("repairing catalog")
                await self.save()
        else:
            logger.info("all detections are ok")

    def prepare(self, csv_header: list[str] | None = None) -> None:
        """Prepare the search run."""
        logger.debug("preparing catalog")
        stats = self.get_stats()
        stats.n_detections = self.n_events
        if self and self.n_events:
            stats.max_semblance = max(detection.semblance for detection in self)
        if csv_header is not None:
            self._csv_header = csv_header

    async def save_detection(
        self,
        detection: EventDetection,
        update: bool = False,
        jitter_location: float = 0.0,
    ) -> None:
        """Dump a detection data to file.

        After the detection is dumped, the receivers are dumped to a separate file and
        the receivers cache is cleared.

        Args:
            detection (EventDetection): The detection to save.
            update (bool): Whether to update an existing detection or append a new one.
            jitter_location (float, optional): The amount of spatial jitter to apply to
                the exported detection. Defaults to 0.0.

        Raises:
            ValueError: If the detection index is not set and update is True.
        """
        json_data = detection.model_dump_json(exclude={"receivers"})

        if update:
            if detection._detection_idx is None:
                raise AttributeError("cannot update detection without set index")
            logger.debug("updating detection %d", detection._detection_idx)

            async with UPDATE_LOCK:
                async with aiofiles.open(self.detections_file, "r") as f:
                    lines = await f.readlines()

                lines[detection._detection_idx] = json_data + "\n"
                async with aiofiles.open(self.detections_file, "w") as f:
                    await asyncio.shield(f.writelines(lines))
            return

        logger.debug("appending detection %d", detection._detection_idx)
        async with UPDATE_LOCK:
            async with aiofiles.open(self.detections_file, "a") as f:
                await f.write(json_data + "\n")
                await f.flush()

            async with aiofiles.open(self.receivers_file, "a") as f:
                await f.write(detection.receivers.model_dump_json() + "\n")
                await f.flush()

            detection.write_csv_line(
                self.csv_dir / "detections.csv",
                header=self._csv_header,
            )
            detection.write_pyrocko_event(
                self.rundir / "pyrocko_detections.list",
            )
            if jitter_location:
                detection.write_csv_line(
                    self.csv_dir / "detections_jittered.csv",
                    header=self._csv_header,
                    jitter_location=jitter_location,
                )
                detection.write_pyrocko_event(
                    self.rundir / "pyrocko_detections_jittered.list",
                    jitter_location=jitter_location,
                )

        detection.clear_receivers()

    async def save(self) -> None:
        """Save catalog to current rundir."""
        logger.debug("saving %d detections", self.n_events)

        lines_events = []
        lines_recv = []
        # Has to be the unsorted
        for detection in self.events:
            lines_events.append(detection.model_dump_json(exclude={"receivers"}) + "\n")
            lines_recv.append(detection.receivers.model_dump_json() + "\n")

        async with UPDATE_LOCK:
            async with aiofiles.open(self.detections_file, "w") as f:
                await f.writelines(lines_events)
            async with aiofiles.open(self.receivers_file, "w") as f:
                await f.writelines(lines_recv)

    async def export_detections(self, jitter_location: float = 0.0) -> None:
        """Export detections to CSV and Pyrocko event lists in the current rundir.

        Args:
            jitter_location (float): The amount of jitter in [m] to apply
                to the detection locations. Defaults to 0.0.
        """
        logger.debug("exporting detections")
        self.sort()

        await self.export_csv(self.csv_dir / "detections.csv")
        self.export_pyrocko_events(self.rundir / "pyrocko_detections.list")

        if jitter_location:
            await self.export_csv(
                self.csv_dir / "detections_jittered.csv",
                jitter_location=jitter_location,
            )
            self.export_pyrocko_events(
                self.rundir / "pyrocko_detections_jittered.list",
                jitter_location=jitter_location,
            )

    async def export_csv(self, file: Path, jitter_location: float = 0.0) -> None:
        """Export detections to a CSV file.

        Args:
            file (Path): The output filename.
            jitter_location (float, optional): Randomize the location of each detection
                by this many meters. Defaults to 0.0.
        """
        logger.info("exporting event CSV to %s", file)
        header = []

        if jitter_location:
            detections = [det.jitter_location(jitter_location) for det in self]
        else:
            detections = self.events

        csv_lines: list[dict] = []
        for detection in detections:
            csv = detection.get_csv_dict()
            for key in csv:
                if key not in header:
                    header.append(key)
            csv_lines.append(csv)

        header = header or self._csv_header
        header_line = [",".join(header) + "\n"]
        rows = [
            ",".join(str(csv.get(key, '""')) for key in header) + "\n"
            for csv in csv_lines
        ]

        async with aiofiles.open(file, "w") as f:
            await f.writelines(header_line + rows)

    def as_pyrocko_events(self) -> list[Event]:
        """Convert the detections to Pyrocko Event objects.

        Returns:
            list[Event]: A list of Pyrocko Event objects.
        """
        return [det.as_pyrocko_event() for det in self.events]

    def get_pyrocko_markers(
        self,
        modelled: bool = True,
        observed: bool = True,
    ) -> list[EventMarker | PhaseMarker]:
        """Get Pyrocko phase pick markers for all detections.

        Args:
            modelled (bool, optional): Include modelled picks. Defaults to True.
            observed (bool, optional): Include observed picks. Defaults to True.

        Returns:
            list[EventMarker | PhaseMarker]: A list of Pyrocko PhaseMarker.
        """
        logger.info("loading Pyrocko markers...")
        pyrocko_markers = []
        for detection in self:
            pyrocko_markers.extend(detection.get_pyrocko_markers(modelled, observed))
        return pyrocko_markers

    def export_pyrocko_events(
        self, filename: Path, jitter_location: float = 0.0
    ) -> None:
        """Export Pyrocko events for all detections to a file.

        Args:
            filename (Path): output filename
            jitter_location (float, optional): Randomize the location of each detection
                by this many meters. Defaults to 0.0.
        """
        logger.info("exporting Pyrocko events to %s", filename)
        detections = self.events
        if jitter_location:
            detections = [det.jitter_location(jitter_location) for det in detections]
        dump_events(
            [det.as_pyrocko_event() for det in detections],
            filename=str(filename),
            format="yaml",
        )

    def export_pyrocko_markers(self, filename: Path) -> None:
        """Export Pyrocko markers for all detections to a file.

        Args:
            filename (Path): output filename
        """
        logger.info("exporting Pyrocko markers to %s", filename)
        pyrocko_markers = []
        for detection in self:
            pyrocko_markers.extend(detection.get_pyrocko_markers())
        marker.save_markers(pyrocko_markers, str(filename))

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(sorted(self.events, key=lambda d: d.time))
