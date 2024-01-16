from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import aiofiles
from pydantic import BaseModel, PrivateAttr
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import dump_events
from pyrocko.trace import Trace
from rich.table import Table

from qseek.console import console
from qseek.models.detection import (
    FILENAME_DETECTIONS,
    FILENAME_RECEIVERS,
    UPDATE_LOCK,
    EventDetection,
    logger,
)
from qseek.stats import Stats
from qseek.utils import Symbols, time_to_path


class EventCatalogStats(Stats):
    n_detections: int = 0
    max_semblance: float = 0.0

    _position: int = 2

    def new_detection(self, detection: EventDetection):
        self.n_detections += 1
        self.max_semblance = max(self.max_semblance, detection.semblance)

    def _populate_table(self, table: Table) -> None:
        table.add_row("No. Detections", f"[bold]{self.n_detections} :dim_button:")
        table.add_row("Maximum semblance", f"{self.max_semblance:.4f}")


class EventCatalog(BaseModel):
    rundir: Path
    detections: list[EventDetection] = []
    _stats: EventCatalogStats = PrivateAttr(default_factory=EventCatalogStats)

    def model_post_init(self, __context: Any) -> None:
        EventDetection.set_rundir(self.rundir)

    @property
    def n_detections(self) -> int:
        """Number of detections"""
        return len(self.detections)

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

    async def add(self, detection: EventDetection) -> None:
        detection.set_index(self.n_detections)

        markers_file = self.markers_dir / f"{time_to_path(detection.time)}.list"
        self.markers_dir.mkdir(exist_ok=True)
        detection.export_pyrocko_markers(markers_file)

        self.detections.append(detection)
        logger.info(
            "%s event detection %d %s: %.5f°, %.5f°, depth %.1f m, "
            "border distance %.1f m, semblance %.3f, magnitude %.2f",
            Symbols.Target,
            self.n_detections,
            detection.time,
            *detection.effective_lat_lon,
            detection.depth,
            detection.distance_border,
            detection.semblance,
            detection.magnitude.average if detection.magnitude else 0.0,
        )
        self._stats.new_detection(detection)
        # This has to happen after the markers are saved, cache is cleared
        await detection.save()

    def save_semblance_trace(self, trace: Trace) -> None:
        """Add semblance trace to detection and save to file.

        Args:
            trace (Trace): semblance trace.
        """
        trace.set_station("SEMBL")
        io.save(
            trace,
            str(self.rundir / "semblance.mseed"),
            append=True,
        )

    @classmethod
    def load_rundir(cls, rundir: Path) -> EventCatalog:
        """Load detections from files in the detections directory."""
        detection_file = rundir / FILENAME_DETECTIONS
        catalog = cls(rundir=rundir)

        if not detection_file.exists():
            logger.warning("cannot find %s", detection_file)
            return catalog

        with console.status(f"loading detections from {rundir}..."), open(
            detection_file
        ) as f:
            for idx, line in enumerate(f):
                detection = EventDetection.model_validate_json(line)
                detection.set_index(idx)
                catalog.detections.append(detection)

        logger.info("loaded %d detections", catalog.n_detections)

        stats = catalog._stats
        stats.n_detections = catalog.n_detections
        if catalog:
            stats.max_semblance = max(detection.semblance for detection in catalog)
        return catalog

    async def save(self) -> None:
        """Save catalog to current rundir."""
        logger.debug("saving %d detections", self.n_detections)

        lines_events = []
        lines_recv = []
        # Has to be the unsorted
        for detection in self.detections:
            lines_events.append(f"{detection.model_dump_json(exclude={'receivers'})}\n")
            lines_recv.append(f"{detection.receivers.model_dump_json()}\n")

        async with UPDATE_LOCK:
            async with aiofiles.open(self.rundir / FILENAME_DETECTIONS, "w") as f:
                await f.writelines(lines_events)
            async with aiofiles.open(self.rundir / FILENAME_RECEIVERS, "w") as f:
                await f.writelines(lines_recv)

    async def export_detections(self, jitter_location: float = 0.0) -> None:
        """
        Export detections to CSV and Pyrocko event lists in the current rundir.

        Args:
            jitter_location (float): The amount of jitter in [m] to apply
                to the detection locations. Defaults to 0.0.
        """
        logger.debug("dumping detections")

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
        header = []

        if jitter_location:
            detections = [det.jitter_location(jitter_location) for det in self]
        else:
            detections = self.detections

        csv_dicts: list[dict] = []
        for detection in detections:
            csv = detection.get_csv_dict()
            for key in csv:
                if key not in header:
                    header.append(key)
            csv_dicts.append(csv)

        header_line = [",".join(header) + "\n"]
        rows = [
            ",".join(str(csv.get(key, "")) for key in header) + "\n"
            for csv in csv_dicts
        ]

        async with aiofiles.open(file, "w") as f:
            await f.writelines(header_line + rows)

    def export_pyrocko_events(
        self, filename: Path, jitter_location: float = 0.0
    ) -> None:
        """Export Pyrocko events for all detections to a file

        Args:
            filename (Path): output filename
        """
        logger.info("saving Pyrocko events to %s", filename)
        detections = self.detections
        if jitter_location:
            detections = [det.jitter_location(jitter_location) for det in detections]
        dump_events(
            [det.as_pyrocko_event() for det in detections],
            filename=str(filename),
            format="yaml",
        )

    def export_pyrocko_markers(self, filename: Path) -> None:
        """Export Pyrocko markers for all detections to a file

        Args:
            filename (Path): output filename
        """
        logger.info("saving Pyrocko markers to %s", filename)
        pyrocko_markers = []
        for detection in self:
            pyrocko_markers.extend(detection.get_pyrocko_markers())
        marker.save_markers(pyrocko_markers, str(filename))

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(sorted(self.detections, key=lambda d: d.time))
