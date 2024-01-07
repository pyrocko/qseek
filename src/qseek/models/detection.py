from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from random import uniform
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator
from uuid import UUID, uuid4

import aiofiles
from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    PositiveFloat,
    PrivateAttr,
    computed_field,
    field_validator,
)
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import Event, dump_events
from pyrocko.squirrel.error import NotAvailable
from rich.table import Table
from typing_extensions import Self

from qseek.console import console
from qseek.features import EventFeaturesType
from qseek.images.images import ImageFunctionPick
from qseek.magnitudes import EventMagnitudeType
from qseek.models.detection_uncertainty import DetectionUncertainty
from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.stats import Stats
from qseek.tracers.tracers import RayTracerArrival
from qseek.utils import (
    NSL,
    MeasurementUnit,
    PhaseDescription,
    Symbols,
    filter_clipped_traces,
    time_to_path,
)

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.features.base import EventFeature
    from qseek.magnitudes.base import EventMagnitude


logger = logging.getLogger(__name__)

FILENAME_DETECTIONS = "detections.json"
FILENAME_RECEIVERS = "detections_receivers.json"

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

    def get_row(self, row_index: int) -> str:
        if self.mtime is None or self.mtime != self.file.stat().st_mtime:
            self.load()
        return self.lines[row_index]


class PhaseDetection(BaseModel):
    phase: PhaseDescription
    model: RayTracerArrival
    observed: ImageFunctionPick | None = None

    @property
    def traveltime_delay(self) -> timedelta | None:
        """Traveltime delay between observed and modelled arrival.

        Returns:
            timedelta | None: The time difference between the observed and modelled arrival,
            or None if the observed time is not available.
        """
        if self.observed:
            return self.observed.time - self.model.time
        return None

    def get_arrival_time(self) -> datetime:
        """Get observed time or modelled time, observed phase has priority.

        Returns:
            datetime: Arrival time
        """
        return self.observed.time if self.observed else self.model.time

    def _get_csv_dict(self) -> dict[str, Any]:
        prefix = self.phase
        csv_dict = {f"{prefix}.model.time": self.model.time}
        if self.observed:
            csv_dict[f"{prefix}.model.time"] = self.observed.time
        if self.traveltime_delay:
            csv_dict[
                f"{prefix}.traveltime_delay"
            ] = self.traveltime_delay.total_seconds()
        return csv_dict

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        """
        Convert the observed and modeled arrivals to a list of Pyrocko PhaseMarkers.

        Returns:
            list[marker.PhaseMarker]: List of Pyrocko PhaseMarker objects representing
                the phase detection.
        """
        phase_picks = [
            marker.PhaseMarker(
                nslc_ids=[()],  # Patched by Receiver.as_pyrocko_marker
                tmax=self.model.time.timestamp(),
                tmin=self.model.time.timestamp(),
                kind=0,
                phasename=f"{self.phase[-1]}mod",
            )
        ]
        if self.observed:
            phase_picks.append(
                marker.PhaseMarker(
                    nslc_ids=[()],  # Patched by Receiver.as_pyrocko_marker
                    tmax=self.observed.time.timestamp(),
                    tmin=self.observed.time.timestamp(),
                    automatic=True,
                    kind=1,
                    phasename=f"{self.phase[-1]}obs",
                )
            )
        return phase_picks


class Receiver(Station):
    phase_arrivals: dict[PhaseDescription, PhaseDetection] = {}

    def add_phase_detection(self, arrival: PhaseDetection) -> None:
        self.phase_arrivals[arrival.phase] = arrival

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        """
        Convert the phase arrivals to Pyrocko markers.

        Returns:
            A list of Pyrocko PhaseMarker objects.
        """
        picks = []
        for pick in chain.from_iterable(
            (arr.as_pyrocko_markers() for arr in self.phase_arrivals.values())
        ):
            pick.nslc_ids = [(*self.nsl, "*")]
            picks.append(pick)
        return picks

    def get_arrivals_time_window(
        self, phase: PhaseDescription | None = None
    ) -> tuple[datetime, datetime]:
        """
        Get the time window for phase arrivals.

        Args:
            phase (PhaseDescription | None): Optional phase description.
                If None, the time window for all arrivals is returned.
                Defaults to None.

        Returns:
            tuple[datetime, datetime]: A tuple containing the start and end time of
                the phase arrivals.
        """
        if phase:
            arrival = self.phase_arrivals[phase]
            start_time = arrival.get_arrival_time()
            return start_time, start_time
        times = [arrival.get_arrival_time() for arrival in self.phase_arrivals.values()]
        return min(times), max(times)

    @classmethod
    def from_station(cls, station: Station) -> Self:
        return cls.model_construct(**station.model_dump())


class EventReceivers(BaseModel):
    event_uid: UUID | None = None
    receivers: list[Receiver] = []

    @property
    def n_receivers(self) -> int:
        """Number of receivers in the receiver set"""
        return len(self.receivers)

    def n_observations(self, phase: PhaseDescription) -> int:
        """Number of observations for a given phase"""
        n_observations = 0
        for receiver in self:
            if (arrival := receiver.phase_arrivals.get(phase)) and arrival.observed:
                n_observations += 1
        return n_observations

    def get_waveforms(
        self,
        squirrel: Squirrel,
        seconds_before: float = 3.0,
        seconds_after: float = 5.0,
        phase: PhaseDescription | None = None,
        receivers: Iterable[Receiver] | None = None,
    ) -> list[Trace]:
        """
        Retrieves and restitutes waveforms for a given squirrel.

        Args:
            squirrel (Squirrel): The squirrel waveform organizer.
            seconds_before (float, optional): Number of seconds before phase arrival
                to retrieve. Defaults to 2.0.
            seconds_after (float, optional): Number of seconds after phase arrival
                to retrieve. Defaults to 5.0.
            phase (PhaseDescription | None, optional): The phase description. If None,
                the whole time window is retrieved. Defaults to None.
            receivers (list[Receiver] | None, optional): The receivers to retrieve
                waveforms for. If None, all receivers are retrieved. Defaults to None.

        Returns:
            list[Trace]: The restituted waveforms.
        """
        receivers = receivers or list(self)
        times = list(
            chain.from_iterable(
                receiver.get_arrivals_time_window(phase) for receiver in receivers
            )
        )
        accessor_id = "qseek.event_detection"

        tmin = min(times).timestamp() - seconds_before
        tmax = max(times).timestamp() + seconds_after
        nslc_ids = [(*receiver.nsl, "*") for receiver in receivers]
        traces = squirrel.get_waveforms(
            codes=nslc_ids,
            tmin=tmin,
            tmax=tmax,
            accessor_id=accessor_id,
            want_incomplete=False,
        )
        squirrel.advance_accessor(accessor_id, cache_id="waveform")

        for tr in traces:
            # Crop to receiver's phase arrival window
            receiver = self.get_receiver(tr.nslc_id[:3])
            tmin, tmax = receiver.get_arrivals_time_window(phase)
            tr.chop(tmin.timestamp() - seconds_before, tmax.timestamp() + seconds_after)

        return traces

    async def get_waveforms_restituted(
        self,
        squirrel: Squirrel,
        seconds_before: float = 2.0,
        seconds_after: float = 5.0,
        seconds_fade: float = 5.0,
        cut_off_fade: bool = True,
        phase: PhaseDescription | None = None,
        quantity: MeasurementUnit = "velocity",
        demean: bool = True,
        remove_clipped: bool = False,
        freqlimits: tuple[float, float, float, float] = (0.01, 0.1, 25.0, 35.0),
        receivers: Iterable[Receiver] | None = None,
    ) -> list[Trace]:
        """
        Retrieves and restitutes waveforms for a given squirrel.

        Args:
            squirrel (Squirrel): The squirrel waveform organizer.
            seconds_before (float, optional): Number of seconds before phase arrival
                to retrieve. Defaults to 2.0.
            seconds_after (float, optional): Number of seconds after phase arrival
                to retrieve. Defaults to 5.0.
            seconds_fade (float, optional): Number of seconds for fade in/out.
                Defaults to 5.0.
            cut_off_fade (bool, optional): Whether to cut off the fade in/out.
                Defaults to True.
            phase (PhaseDescription | None, optional): The phase description. If None,
                the whole time window is retrieved. Defaults to None.
            quantity (MeasurementUnit, optional): The measurement unit.
                Defaults to "velocity".
            demean (bool, optional): Whether to demean the waveforms. Defaults to True.
            remove_clipped (bool, optional): Whether to remove clipped traces.
                Defaults to False.
            freqlimits (tuple[float, float, float, float], optional):
                The frequency limits. Defaults to (0.01, 0.1, 25.0, 35.0).
            receivers (list[Receiver] | None, optional): The receivers to retrieve
                waveforms for. If None, all receivers are retrieved. Defaults to None.

        Returns:
            list[Trace]: The restituted waveforms.
        """
        traces = await asyncio.to_thread(
            self.get_waveforms,
            squirrel,
            phase=phase,
            seconds_after=seconds_after + seconds_fade,
            seconds_before=seconds_before + seconds_fade,
            receivers=receivers,
        )
        traces = filter_clipped_traces(traces) if remove_clipped else traces

        restituted_traces = []
        for tr in traces:
            try:
                response = squirrel.get_response(
                    tmin=tr.tmin,
                    tmax=tr.tmax,
                    codes=[tr.nslc_id],
                )
            except NotAvailable:
                continue

            restituted_traces.append(
                await asyncio.to_thread(
                    tr.transfer,
                    transfer_function=response.get_effective(input_quantity=quantity),
                    freqlimits=freqlimits,
                    tfade=seconds_fade,
                    cut_off_fading=cut_off_fade,
                    demean=demean,
                    invert=True,
                )
            )
        return restituted_traces

    def get_receiver(self, nsl: NSL) -> Receiver:
        """
        Get the receiver object based on given NSL tuple.

        Args:
            nsl (tuple[str, str, str]): The network, station, and location tuple.

        Returns:
            Receiver: The receiver object matching the given nsl tuple.

        Raises:
            KeyError: If no receiver is found with the given nsl tuple.
        """
        for receiver in self:
            if receiver.nsl == nsl:
                return receiver
        raise KeyError(f"cannot find station {nsl.pretty}")

    def add(
        self,
        stations: Stations,
        phase_arrivals: list[PhaseDetection | None],
    ) -> None:
        """Add receivers to the receiver set

        Args:
            stations: List of stations
            phase_arrivals: List of phase arrivals

        Raises:
            KeyError: If a station is not found in the receiver set
        """
        receivers = [Receiver.from_station(sta) for sta in stations]
        for receiver, arrival in zip(receivers, phase_arrivals, strict=True):
            if not arrival:
                continue
            try:
                receiver = self.get_by_nsl(receiver.nsl)
            except KeyError:
                self.receivers.append(receiver)
            receiver.add_phase_detection(arrival)

    def get_by_nsl(self, nsl: NSL) -> Receiver:
        """
        Retrieves a receiver object by its NSL (network, station, location) tuple.

        Args:
            nsl (NSL): The NSL tuple representing
                the network, station, and location.

        Returns:
            Receiver: The receiver object matching the NSL tuple.

        Raises:
            KeyError: If no receiver is found with the specified NSL tuple.
        """
        for receiver in self:
            if receiver.nsl == nsl:
                return receiver
        raise KeyError(f"cannot find station {nsl.pretty}")

    def get_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        """
        Get a list of Pyrocko phase markers from all receivers.

        Returns:
            A list of Pyrocko phase markers.
        """
        return list(
            chain.from_iterable((receiver.as_pyrocko_markers() for receiver in self))
        )

    def __iter__(self) -> Iterator[Receiver]:
        return iter(self.receivers)


class EventDetection(Location):
    uid: UUID = Field(default_factory=uuid4)

    time: AwareDatetime = Field(
        ...,
        description="Detection time",
    )
    semblance: PositiveFloat = Field(
        ...,
        description="Detection semblance",
    )
    n_stations: int = Field(
        default=0,
        description="Number of stations in the detection.",
    )

    distance_border: PositiveFloat = Field(
        ...,
        description="Distance to the nearest border in meters. "
        "Only distance to NW, SW and bottom border is considered.",
    )
    in_bounds: bool = Field(
        default=True,
        description="Is detection in bounds, and inside the configured border.",
    )
    uncertainty: DetectionUncertainty | None = Field(
        default=None,
        description="Detection uncertainty.",
    )

    magnitudes: list[EventMagnitudeType] = Field(
        default=[],
        description="Event magnitudes.",
    )
    features: list[EventFeaturesType] = Field(
        default=[],
        description="Event features.",
    )

    _receivers: EventReceivers | None = PrivateAttr(None)

    _detection_idx: int | None = PrivateAttr(None)
    _rundir: ClassVar[Path | None] = None
    _receiver_cache: ClassVar[ReceiverCache | None] = None

    @field_validator("features", mode="before")
    @classmethod
    def migrate_features(cls, v: Any) -> list[EventFeaturesType]:
        # FIXME: Remove this migration
        if isinstance(v, dict):
            return v.get("features", [])
        return v

    @classmethod
    def set_rundir(cls, rundir: Path) -> None:
        """
        Set the rundir for the detection model.

        Args:
            rundir (Path): The path to the rundir.
        """
        cls._rundir = rundir
        cls._receiver_cache = ReceiverCache(rundir / FILENAME_RECEIVERS)

    @property
    def magnitude(self) -> EventMagnitude | None:
        """
        Returns the magnitude of the event.

        If there are no magnitudes available, returns None.
        """
        return self.magnitudes[0] if self.magnitudes else None

    async def save(self, file: Path | None = None, update: bool = False) -> None:
        """
        Dump the detection data to a file.

        After the detection is dumped, the receivers are dumped to a separate file and
        the receivers cache is cleared.

        Args:
            directory (Path): The directory where the file will be saved.
            update (bool): Whether to update an existing detection or append a new one.

        Raises:
            ValueError: If the detection index is not set and update is True.
        """
        if not file and self._rundir:
            file = self._rundir / FILENAME_DETECTIONS
        else:
            raise ValueError("cannot dump detection without set rundir")

        json_data = self.model_dump_json(exclude={"receivers"})

        if update:
            if self._detection_idx is None:
                raise ValueError("cannot update detection without set index")
            logger.debug("updating detection %d", self._detection_idx)

            async with UPDATE_LOCK:
                async with aiofiles.open(file, "r") as f:
                    lines = await f.readlines()

                lines[self._detection_idx] = f"{json_data}\n"
                async with aiofiles.open(file, "w") as f:
                    await asyncio.shield(f.writelines(lines))
        else:
            logger.debug("appending detection %d", self._detection_idx)
            async with aiofiles.open(file, "a") as f:
                await f.write(f"{json_data}\n")

            receiver_file = self._rundir / FILENAME_RECEIVERS
            async with aiofiles.open(receiver_file, "a") as f:
                await asyncio.shield(f.write(f"{self.receivers.model_dump_json()}\n"))

            self._receivers = None  # Free the memory

    def set_index(self, index: int) -> None:
        """
        Set the index of the detection.

        Args:
            index (int): The index to set.

        Returns:
            None
        """
        if self._detection_idx is not None:
            raise ValueError("cannot set index twice")
        self._detection_idx = index

    def set_uncertainty(self, uncertainty: DetectionUncertainty) -> None:
        """Set detection uncertainty

        Args:
            uncertainty (DetectionUncertainty): detection uncertainty
        """
        self.uncertainty = uncertainty

    def add_magnitude(self, magnitude: EventMagnitude) -> None:
        """Add magnitude to detection

        Args:
            magnitude (EventMagnitudeType): magnitude
        """
        self.magnitudes.append(magnitude)

    def add_feature(self, feature: EventFeature) -> None:
        """Add feature to the feature set.

        Args:
            feature (EventFeature): Feature to add
        """
        self.features.append(feature)

    @computed_field
    @property
    def receivers(self) -> EventReceivers:
        """
        Retrieves the event receivers associated with the detection.

        Returns:
            EventReceivers: The event receivers associated with the detection.

        Raises:
            ValueError: If the receivers cannot be fetched due to missing rundir
                        and index, or if there is a UID mismatch between the fetched
                        receivers and the detection.
        """
        if self._receivers is not None:
            ...
        elif self._detection_idx is None:
            self._receivers = EventReceivers(event_uid=self.uid)
        elif self._rundir and self._detection_idx is not None:
            if self._receiver_cache is None:
                raise ValueError("cannot fetch receivers without set rundir")

            logger.debug("fetching receiver information from cache")
            row = self._receiver_cache.get_row(self._detection_idx)
            receivers = EventReceivers.model_validate_json(row)

            if receivers.event_uid != self.uid:
                raise ValueError(f"uid mismatch: {receivers.event_uid} != {self.uid}")
            self._receivers = receivers
        else:
            raise ValueError("cannot fetch receivers without set rundir and index")
        return self._receivers

    def as_pyrocko_event(self) -> Event:
        """Get detection as Pyrocko event

        Returns:
            Event: Pyrocko event
        """
        magnitude = self.magnitude
        return Event(
            name=self.time.isoformat(sep="T"),
            time=self.time.timestamp(),
            lat=self.lat,
            lon=self.lon,
            east_shift=self.east_shift,
            north_shift=self.north_shift,
            depth=self.effective_depth,
            magnitude=magnitude.average if magnitude else self.semblance,
            magnitude_type=magnitude.name if magnitude else "semblance",
            extras={"semblance": self.semblance},
        )

    def get_csv_dict(self) -> dict[str, Any]:
        """Get detection as CSV line

        Returns:
            dict[str, Any]: CSV line
        """
        csv_line = {
            "time": self.time,
            "lat": round(self.effective_lat, 6),
            "lon": round(self.effective_lon, 6),
            "depth": round(self.effective_depth, 2),
            "east_shift": round(self.east_shift, 2),
            "north_shift": round(self.north_shift, 2),
            "distance_border": round(self.distance_border, 2),
            "in_bounds": self.in_bounds,
            "semblance": self.semblance,
        }
        for magnitude in self.magnitudes:
            csv_line.update(magnitude.csv_row())
        return csv_line

    def get_pyrocko_markers(self) -> list[marker.EventMarker | marker.PhaseMarker]:
        """Get detections as Pyrocko markers

        Returns:
            list[marker.EventMarker | marker.PhaseMarker]: Pyrocko markers
        """
        event = self.as_pyrocko_event()

        pyrocko_markers: list[marker.EventMarker | marker.PhaseMarker] = [
            marker.EventMarker(event)
        ]
        for phase_pick in self.receivers.get_pyrocko_markers():
            phase_pick.set_event(event)
            pyrocko_markers.append(phase_pick)
        return pyrocko_markers

    def export_pyrocko_markers(self, filename: Path) -> None:
        """Save detection's Pyrocko markers to file

        Args:
            filename (Path): path to marker file
        """
        logger.debug("dumping detection's Pyrocko markers to %s", filename)
        marker.save_markers(self.get_pyrocko_markers(), str(filename))

    def jitter_location(self, meters: float) -> Self:
        """Randomize detection location

        Args:
            meters (float): maximum randomization in meters

        Returns:
            EventDetection: spatially jittered detection
        """
        half_meters = meters / 2
        detection = self.model_copy()
        detection.east_shift += uniform(-half_meters, half_meters)
        detection.north_shift += uniform(-half_meters, half_meters)
        detection.depth += uniform(-half_meters, half_meters)
        detection._cached_lat_lon = None
        return detection

    def snuffle(self, squirrel: Squirrel, restituted: bool = False) -> None:
        """Open snuffler for detection

        Args:
            squirrel (Squirrel): The squirrel, holding the data
            restituted (bool, optional): Restitude the data. Defaults to False.
        """
        from pyrocko.trace import snuffle

        traces = (
            self.receivers.get_waveforms(squirrel)
            if not restituted
            else self.receivers.get_waveforms_restituted(squirrel)
        )
        snuffle(
            traces,
            markers=self.get_pyrocko_markers(),
            stations=[recv.as_pyrocko_station() for recv in self.receivers],
        )

    def __str__(self) -> str:
        # TODO: Add more information
        return str(self.time)


class DetectionsStats(Stats):
    n_detections: int = 0
    max_semblance: float = 0.0

    _position: int = 2

    def new_detection(self, detection: EventDetection):
        self.n_detections += 1
        self.max_semblance = max(self.max_semblance, detection.semblance)

    def _populate_table(self, table: Table) -> None:
        table.add_row("No. Detections", f"[bold]{self.n_detections} :dim_button:")
        table.add_row("Maximum semblance", f"{self.max_semblance:.4f}")


class EventDetections(BaseModel):
    rundir: Path
    detections: list[EventDetection] = []
    _stats: DetectionsStats = PrivateAttr(default_factory=DetectionsStats)

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
    def load_rundir(cls, rundir: Path) -> EventDetections:
        """Load detections from files in the detections directory."""
        detection_file = rundir / FILENAME_DETECTIONS
        detections = cls(rundir=rundir)

        if not detection_file.exists():
            logger.warning("cannot find %s", detection_file)
            return detections

        with console.status(f"loading detections from {rundir}..."), open(
            detection_file
        ) as f:
            for idx, line in enumerate(f):
                detection = EventDetection.model_validate_json(line)
                detection.set_index(idx)
                detections.detections.append(detection)

        logger.info("loaded %d detections", detections.n_detections)

        stats = detections._stats
        stats.n_detections = detections.n_detections
        if detections:
            stats.max_semblance = max(detection.semblance for detection in detections)
        return detections

    async def save(self) -> None:
        """Save detections to current rundir."""
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
