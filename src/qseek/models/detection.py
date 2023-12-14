from __future__ import annotations

import logging
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from random import uniform
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal, Type, TypeVar
from uuid import UUID, uuid4

import numpy as np
from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    PositiveFloat,
    PrivateAttr,
    computed_field,
)
from pyevtk.hl import pointsToVTK
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import Event, dump_events
from rich.table import Table
from typing_extensions import Self

from qseek.console import console
from qseek.features import EventFeaturesTypes, ReceiverFeaturesTypes
from qseek.images.images import ImageFunctionPick
from qseek.models.detection_uncertainty import DetectionUncertainty
from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.stats import Stats
from qseek.tracers.tracers import RayTracerArrival
from qseek.utils import PhaseDescription, Symbols, time_to_path

if TYPE_CHECKING:
    from pyrocko.squirrel import Response, Squirrel
    from pyrocko.trace import Trace


_ReceiverFeature = TypeVar("_ReceiverFeature", bound=ReceiverFeaturesTypes)
_EventFeature = TypeVar("_EventFeature", bound=EventFeaturesTypes)
logger = logging.getLogger(__name__)


MeasurementUnit = Literal[
    # "displacement",
    "velocity",
    "acceleration",
]


FILENAME_DETECTIONS = "detections.json"
FILENAME_RECEIVERS = "detections_receivers.json"


class PhaseDetection(BaseModel):
    phase: PhaseDescription
    model: RayTracerArrival
    observed: ImageFunctionPick | None = None

    @property
    def traveltime_delay(self) -> timedelta | None:
        """Traveltime delay between observed and modelled arrival."""
        if self.observed:
            return self.observed.time - self.model.time
        return None

    def get_arrival_time(self) -> datetime:
        """Get observed time or modelled time, if observed is not set.

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
    features: list[ReceiverFeaturesTypes] = []
    phase_arrivals: dict[PhaseDescription, PhaseDetection] = {}

    def add_phase_detection(self, arrival: PhaseDetection) -> None:
        self.phase_arrivals[arrival.phase] = arrival

    def add_feature(self, feature: ReceiverFeaturesTypes) -> None:
        for existing_feature in self.features.copy():
            if isinstance(feature, existing_feature.__class__):
                logger.debug("replacing existing feature %s", feature.feature)
                self.features.remove(existing_feature)
                break
        self.features.append(feature)

    def get_feature(self, feature_type: Type[_ReceiverFeature]) -> _ReceiverFeature:
        for feature in self.features:
            if isinstance(feature, feature_type):
                return feature
        raise TypeError(f"cannot find feature of type {feature_type.__class__}")

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        picks = []
        for pick in chain.from_iterable(
            (arr.as_pyrocko_markers() for arr in self.phase_arrivals.values())
        ):
            pick.nslc_ids = [(*self.nsl, "*")]
            picks.append(pick)
        return picks

    def get_arrival_time_range(
        self, phase: PhaseDescription | None = None
    ) -> tuple[datetime, datetime]:
        if phase:
            arrival = self.phase_arrivals[phase]
            start_time = arrival.get_arrival_time()
            return start_time, start_time
        times = [arrival.get_arrival_time() for arrival in self.phase_arrivals.values()]
        return min(times), max(times)

    def get_waveforms(
        self,
        squirrel: Squirrel,
        seconds_after: float = 5.0,
        seconds_before: float = 3.0,
        phase: PhaseDescription | None = None,
    ) -> list[Trace]:
        if phase:
            arrival = self.phase_arrivals[phase]
            start_time = arrival.get_arrival_time()
            end_time = start_time
        else:
            start_time, end_time = self.get_arrival_time_range()

        traces = squirrel.get_waveforms(
            codes=[(*self.nsl, "*")],
            tmin=(start_time - timedelta(seconds=seconds_before)).timestamp(),
            tmax=(end_time + timedelta(seconds=seconds_after)).timestamp(),
            want_incomplete=False,
        )
        if not traces:
            raise KeyError
        return traces

    def get_waveforms_restituted(
        self,
        squirrel: Squirrel,
        seconds_after: float = 5.0,
        seconds_before: float = 2.0,
        phase: PhaseDescription | None = None,
        quantity: MeasurementUnit = "velocity",
        demean: bool = True,
        # TODO: Improve freqlimits
        freqlimits: tuple[float, float, float, float] = (0.01, 0.1, 25.0, 35.0),
    ) -> list[Trace]:
        traces = self.get_waveforms(
            squirrel,
            phase=phase,
            seconds_after=seconds_after,
            seconds_before=seconds_before,
        )

        restituted_traces = []
        for tr in traces:
            response: Response = squirrel.get_response(
                tmin=tr.tmin, tmax=tr.tmax, codes=[tr.nslc_id]
            )

            # TODO: Add more padding when displacement is requested, use Trace.tfade
            # Get waveforms provides more data
            restituted_traces.append(
                tr.transfer(
                    transfer_function=response.get_effective(input_quantity=quantity),
                    freqlimits=freqlimits,
                    invert=True,
                    demean=demean,
                )
            )
        return restituted_traces

    @classmethod
    def from_station(cls, station: Station) -> Self:
        return cls(**station.model_dump())


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

    def add_receivers(
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

    def get_by_nsl(self, nsl: tuple[str, str, str]) -> Receiver:
        for receiver in self:
            if receiver.nsl == nsl:
                return receiver
        raise KeyError(f"cannot find station {nsl}")

    def get_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        return list(
            chain.from_iterable((receiver.as_pyrocko_markers() for receiver in self))
        )

    def __iter__(self) -> Iterator[Receiver]:
        return iter(self.receivers)


class EventFeatures(BaseModel):
    event_uid: UUID | None = None
    features: list[EventFeaturesTypes] = []

    @property
    def n_features(self) -> int:
        """Number of features in the feature set"""
        return len(self.features)

    def add_feature(self, feature: EventFeaturesTypes) -> None:
        """Add feature to the feature set.

        Args:
            feature (EventFeature): Feature to add
        """
        for existing_feature in self.features.copy():
            if isinstance(feature, existing_feature.__class__):
                logger.debug("replacing existing feature %s", feature.feature)
                self.features.remove(existing_feature)
                break
        self.features.append(feature)

    def get_feature(self, feature_type: Type[_EventFeature]) -> _EventFeature:
        """Retrieve feature from detection

        Args:
            feature_type (Type[_EventFeature]): Feature type to retrieve

        Raises:
            TypeError: If feature type is not found

        Returns:
            EventFeature: The feature
        """
        for feature in self.features:
            if isinstance(feature, feature_type):
                return feature
        raise TypeError(f"cannot find feature of type {feature_type.__class__}")


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
    distance_border: PositiveFloat = Field(
        ...,
        description="Distance to the nearest border in meters. "
        "Only distance to NW, SW and bottom border is considered.",
    )

    in_bounds: bool = Field(
        default=True,
        description="Is detection in bounds, and inside the configured border.",
    )

    magnitude: float | None = Field(
        default=None,
        description="Detection magnitude or semblance.",
    )
    magnitude_type: str | None = Field(
        default=None,
        description="Magnitude type.",
    )

    n_stations: int = Field(
        default=0,
        description="Number of stations in the detection.",
    )

    uncertainty: DetectionUncertainty | None = None
    features: EventFeatures = EventFeatures()

    _receivers: EventReceivers | None = PrivateAttr(None)

    _detection_idx: int | None = PrivateAttr(None)
    _rundir: ClassVar[Path | None] = None

    def model_post_init(self, __context: Any) -> None:
        self.features.event_uid = self.uid

    def dump_append(self, directory: Path, detection_index: int) -> None:
        logger.debug("dumping event, receivers and features to %s", directory)

        event_file = directory / FILENAME_DETECTIONS
        json_data = self.model_dump_json(exclude={"receivers"})
        with event_file.open("a") as f:
            f.write(f"{json_data}\n")

        receiver_file = directory / FILENAME_RECEIVERS
        with receiver_file.open("a") as f:
            f.write(f"{self.receivers.model_dump_json()}\n")

        self._detection_idx = detection_index
        self._receivers = None  # Free the memory

    @computed_field
    @property
    def receivers(self) -> EventReceivers:
        if self._receivers is not None:
            ...
        elif self._detection_idx is None:
            self._receivers = EventReceivers(event_uid=self.uid)
        elif self._rundir and self._detection_idx is not None:
            logger.debug("fetching receiver information from file")
            feature_file = self._rundir / FILENAME_RECEIVERS
            with feature_file.open() as f:
                [next(f) for _ in range(self._detection_idx)]
                receivers = EventReceivers.model_validate_json(next(f))

            if receivers.event_uid != self.uid:
                raise ValueError(f"uid mismatch: {receivers.event_uid} != {self.uid}")
            self._receivers = receivers
        else:
            raise ValueError("cannot fetch receivers without set rundir and index")
        return self._receivers

    def as_pyrocko_event(self) -> Event:
        """Get detection as Pyrocko event"""
        return Event(
            name=self.time.isoformat(sep="T"),
            time=self.time.timestamp(),
            lat=self.lat,
            lon=self.lon,
            east_shift=self.east_shift,
            north_shift=self.north_shift,
            depth=self.effective_depth,
            magnitude=self.magnitude or self.semblance,
            magnitude_type=self.magnitude_type,
        )

    def get_pyrocko_markers(self) -> list[marker.EventMarker | marker.PhaseMarker]:
        """Get detections as Pyrocko markers"""
        event = self.as_pyrocko_event()

        pyrocko_markers: list[marker.EventMarker | marker.PhaseMarker] = [
            marker.EventMarker(event)
        ]
        for phase_pick in self.receivers.get_pyrocko_markers():
            phase_pick.set_event(event)
            pyrocko_markers.append(phase_pick)
        return pyrocko_markers

    def dump_pyrocko_markers(self, filename: Path) -> None:
        """Save detection's Pyrocko markers to file

        Args:
            filename (Path): path to marker file
        """
        logger.info("dumping detection's Pyrocko markers to %s", filename)
        marker.save_markers(self.get_pyrocko_markers(), str(filename))

    def jitter_location(self, meters: float) -> EventDetection:
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
        from pyrocko.trace import snuffle

        if restituted:
            traces = (
                receiver.get_waveforms_restituted(squirrel)
                for receiver in self.receivers
            )
        else:
            traces = (receiver.get_waveforms(squirrel) for receiver in self.receivers)
        snuffle([*chain.from_iterable(traces)], markers=self.get_pyrocko_markers())

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
        EventDetection._rundir = self.rundir

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

    @property
    def vtk_dir(self) -> Path:
        dir = self.rundir / "vtk"
        dir.mkdir(exist_ok=True)
        return dir

    def add(self, detection: EventDetection) -> None:
        markers_file = self.markers_dir / f"{time_to_path(detection.time)}.list"
        self.markers_dir.mkdir(exist_ok=True)
        marker.save_markers(detection.get_pyrocko_markers(), str(markers_file))

        self.detections.append(detection)
        logger.info(
            "%s event detection #%d %s: %.5f°, %.5f°, depth %.1f m, "
            "border distance %.1f m, semblance %.3f",
            Symbols.Target,
            self.n_detections,
            detection.time,
            *detection.effective_lat_lon,
            detection.depth,
            detection.distance_border,
            detection.semblance,
        )
        self._stats.new_detection(detection)
        # This has to happen after the markers are saved
        detection.dump_append(self.rundir, self.n_detections - 1)

    def dump_detections(self, jitter_location: float = 0.0) -> None:
        """Dump all detections to files in the detection directory."""

        logger.debug("dumping detections")
        self.export_csv(self.csv_dir / "detections.csv")
        self.export_pyrocko_events(self.rundir / "pyrocko_detections.list")

        self.export_vtk(self.vtk_dir / "detections")

        if jitter_location:
            self.export_csv(
                self.csv_dir / "detections_jittered.csv",
                jitter_location=jitter_location,
            )
            self.export_pyrocko_events(
                self.rundir / "pyrocko_detections_jittered.list",
                jitter_location=jitter_location,
            )
            self.export_vtk(
                self.vtk_dir / "detections_jittered",
                jitter_location=jitter_location,
            )

    def add_semblance_trace(self, trace: Trace) -> None:
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
            for i_detection, line in enumerate(f):
                detection = EventDetection.model_validate_json(line)
                detection._detection_idx = i_detection
                detections.detections.append(detection)

        console.log(f"loaded {detections.n_detections} detections")
        detections._stats.n_detections = detections.n_detections
        detections._stats.max_semblance = max(
            detection.semblance for detection in detections
        )
        return detections

    def export_csv(self, file: Path, jitter_location: float = 0.0) -> None:
        """Export detections to a CSV file

        Args:
            file (Path): output filename
            randomize_meters (float, optional): randomize the location of each detection
                by this many meters. Defaults to 0.0.
        """
        lines = ["lat,lon,depth,semblance,time,distance_border"]
        for detection in self:
            if jitter_location:
                detection = detection.jitter_location(jitter_location)
            lat, lon = detection.effective_lat_lon
            lines.append(
                f"{lat:.5f},{lon:.5f},{detection.effective_depth:.1f},"
                f" {detection.semblance},{detection.time},{detection.distance_border}"
            )
        file.write_text("\n".join(lines))

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

    def export_vtk(
        self,
        filename: Path,
        jitter_location: float = 0.0,
    ) -> None:
        """Export events as vtk file

        Args:
            filename (Path): output filename, without file extension.
            reference (Location): Relative to this location.
        """
        detections = self.detections
        if jitter_location:
            detections = [det.jitter_location(jitter_location) for det in detections]
        offsets = np.array(
            [(det.east_shift, det.north_shift, det.depth) for det in detections]
        )
        pointsToVTK(
            str(filename),
            np.array(offsets[:, 0]),
            np.array(offsets[:, 1]),
            -np.array(offsets[:, 2]),
            data={
                "semblance": np.array([det.semblance for det in detections]),
                "time": np.array([det.time.timestamp() for det in detections]),
            },
        )

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(sorted(self.detections, key=lambda d: d.time))
