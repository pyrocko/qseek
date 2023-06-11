from __future__ import annotations

import logging
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Self, Type, TypeVar
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.model import Event, dump_events

from lassie.features import EventFeatures, ReceiverFeatures
from lassie.images import ImageFunctionPick
from lassie.models.location import Location
from lassie.models.station import Station
from lassie.plot.octree import plot_octree, plot_octree_surface
from lassie.tracers import RayTracerArrival
from lassie.utils import PhaseDescription, time_to_path

if TYPE_CHECKING:
    from pyrocko.squirrel import Response, Squirrel
    from pyrocko.trace import Trace


_ReceiverFeature = TypeVar("_ReceiverFeature", bound=ReceiverFeatures)
_EventFeature = TypeVar("_EventFeature", bound=EventFeatures)
logger = logging.getLogger(__name__)


MeasurementUnit = Literal[
    # "displacement",
    "velocity",
    "acceleration",
]


class PhaseDetection(BaseModel):
    phase: PhaseDescription
    model: RayTracerArrival
    observed: ImageFunctionPick | None = None

    @property
    def traveltime_delay(self) -> timedelta | None:
        if self.observed:
            return self.model.time - self.observed.time

    def get_arrival_time(self) -> datetime:
        """Return observed time or modelled time, if observed is not set.

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
    features: list[ReceiverFeatures] = []
    phase_arrivals: dict[PhaseDescription, PhaseDetection] = {}

    def add_phase_detection(self, arrival: PhaseDetection) -> None:
        self.phase_arrivals[arrival.phase] = arrival

    def add_feature(self, feature: ReceiverFeatures) -> None:
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
        # TODO: Make freqlimits better
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

    def _get_csv_dict(self) -> dict[str, Any]:
        csv_dict = self.dict(exclude={"features", "phase_arrivals"})
        for arrival in self.phase_arrivals.values():
            csv_dict.update(arrival._get_csv_dict())
        return csv_dict

    @classmethod
    def from_station(cls, station: Station) -> Self:
        return cls(**station.dict())


class Receivers(BaseModel):
    __root__: list[Receiver] = []

    @property
    def n_receivers(self) -> int:
        return len(self.__root__)

    def add_receivers(
        self, stations: list[Station], phase_arrivals: list[PhaseDetection]
    ) -> None:
        receivers = [Receiver.from_station(sta) for sta in stations]
        for receiver, arrival in zip(receivers, phase_arrivals, strict=True):
            try:
                receiver = self.get_by_nsl(receiver.nsl)
            except KeyError:
                self.__root__.append(receiver)
            receiver.add_phase_detection(arrival)

    def get_by_nsl(self, nsl: tuple[str, str, str]) -> Receiver:
        for receiver in self:
            if receiver.nsl == nsl:
                return receiver
        raise KeyError(f"cannot find station {nsl}")

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        return list(
            chain.from_iterable((receiver.as_pyrocko_markers() for receiver in self))
        )

    def save_csv(self, filename: Path) -> None:
        for receiver in self:
            receiver._get_csv_dict()

    def snuffle(self, squirrel: Squirrel, restituted: bool = False) -> None:
        from pyrocko.trace import snuffle

        if restituted:
            traces = (receiver.get_waveforms_restituted(squirrel) for receiver in self)
        else:
            traces = (receiver.get_waveforms(squirrel) for receiver in self)
        snuffle([*chain.from_iterable(traces)])

    def __iter__(self) -> Iterator[Receiver]:
        return iter(self.__root__)


class EventDetection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    semblance: float
    distance_border: float

    magnitude: float | None = None
    magnitude_type: str | None = None

    receivers: Receivers = Receivers()
    features: list[EventFeatures] = []

    def add_feature(self, feature: EventFeatures) -> None:
        for existing_feature in self.features.copy():
            if isinstance(feature, existing_feature.__class__):
                logger.debug("replacing existing feature %s", feature.feature)
                self.features.remove(existing_feature)
                break
        self.features.append(feature)

    def get_feature(self, feature_type: Type[_EventFeature]) -> _EventFeature:
        for feature in self.features:
            if isinstance(feature, feature_type):
                return feature
        raise TypeError(f"cannot find feature of type {feature_type.__class__}")

    def as_pyrocko_event(self) -> Event:
        return Event(
            name=self.time.isoformat(sep="T"),
            time=self.time.timestamp(),
            lat=self.lat,
            lon=self.lon,
            east_shift=self.east_shift,
            north_shift=self.north_shift,
            depth=self.depth,
            elevation=self.elevation,
            magnitude=self.magnitude,
            magnitude_type=self.magnitude_type,
        )

    def as_pyrocko_markers(self) -> list[marker.EventMarker | marker.PhaseMarker]:
        event = self.as_pyrocko_event()

        pyrocko_markers: list[marker.EventMarker | marker.PhaseMarker] = [
            marker.EventMarker(event)
        ]
        for phase_pick in self.receivers.as_pyrocko_markers():
            phase_pick.set_event(event)
            pyrocko_markers.append(phase_pick)
        return pyrocko_markers

    def save_pyrocko_markers(self, filename: Path) -> None:
        logger.info("saving detection's Pyrocko markers to %s", filename)
        marker.save_markers(self.as_pyrocko_markers(), str(filename))

    def plot(self, cmap: str = "Oranges") -> None:
        plot_octree(self.octree, cmap=cmap)

    def plot_surface(
        self, accumulator: Callable = np.max, cmap: str = "Oranges"
    ) -> None:
        plot_octree_surface(self.octree, accumulator=accumulator, cmap=cmap)


class Detections(BaseModel):
    rundir: Path
    detections: list[EventDetection] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.detections_dir.exists():
            self.detections_dir.mkdir()
            logger.info("created directory %s", self.detections_dir)
        else:
            logger.info("loading detections from %s", self.detections_dir)
            self.load_detections()

    @property
    def detections_dir(self) -> Path:
        return self.rundir / "detections"

    def add(self, detection: EventDetection) -> None:
        # detection.octree.make_concrete()
        self.detections.append(detection)

        filename = self.detections_dir / (time_to_path(detection.time) + ".json")
        filename.write_text(detection.json())

        self.save_csv(self.rundir / "detections.csv")
        self.save_pyrocko_events(self.rundir / "pyrocko-events.list")
        self.save_pyrocko_markers(self.rundir / "pyrocko-markers.list")

    def add_semblance(self, trace: Trace) -> None:
        trace.set_station("SEMBL")
        io.save(
            trace,
            str(self.rundir / "semblance.mseed"),
            append=True,
        )

    def load_detections(self) -> None:
        logger.info("loading detections from %s", self.detections_dir)
        for file in sorted(self.detections_dir.glob("*.json")):
            detection = EventDetection.parse_file(file)
            self.detections.append(detection)

    def get(self, uid: UUID) -> EventDetection:
        for detection in self:
            if detection.uid == uid:
                return detection
        raise KeyError("detection not found")

    def n_detections(self) -> int:
        return len(self.detections)

    def save_csv(self, file: Path) -> None:
        lines = ["lat, lon, depth, detection_peak, time"]
        for det in self:
            lat, lon = det.effective_lat_lon
            lines.append(
                f"{lat:.5f}, {lon:.5f}, {-det.effective_elevation:.1f},"
                f" {det.semblance}, {det.time}"
            )
        file.write_text("\n".join(lines))

    def save_pyrocko_events(self, filename: Path) -> None:
        logger.info("saving Pyrocko events to %s", filename)
        dump_events(
            [detection.as_pyrocko_event() for detection in self],
            filename=str(filename),
        )

    def save_pyrocko_markers(self, filename: Path) -> None:
        logger.info("saving Pyrocko markers to %s", filename)
        pyrocko_markers = []
        for detection in self:
            pyrocko_markers.extend(detection.as_pyrocko_markers())
        marker.save_markers(pyrocko_markers, str(filename))

    def __iter__(self) -> Iterator[EventDetection]:
        return iter(self.detections)
