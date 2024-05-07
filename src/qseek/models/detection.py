from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from functools import cached_property
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
from pyrocko.gui import marker
from pyrocko.model import Event
from typing_extensions import Self

from qseek.features import EventFeaturesType
from qseek.images.base import ObservedArrival
from qseek.magnitudes import EventMagnitudeType
from qseek.models.detection_uncertainty import DetectionUncertainty
from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.tracers.base import ModelledArrival
from qseek.utils import (
    NSL,
    MeasurementUnit,
    PhaseDescription,
    filter_clipped_traces,
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
SQUIRREL_SEM = asyncio.Semaphore(64)


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

    def find_uid(self, uid: UUID) -> tuple[int, str]:
        """Find the given UID in the lines and return its index and value.

        get_line should be prefered over this method.

        Args:
            uid (UUID): The UID to search for.

        Returns:
            tuple[int, str]: A tuple containing the index and value of the found UID.
        """
        self._check_mtime()
        find_uid = str(uid)
        for iline, line in enumerate(self.lines):
            if find_uid in line:
                return iline, line
        raise KeyError


class PhaseDetection(BaseModel):
    phase: PhaseDescription
    model: ModelledArrival
    observed: ObservedArrival | None = None
    station_delay: timedelta | None = None

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
            csv_dict[f"{prefix}.traveltime_delay"] = (
                self.traveltime_delay.total_seconds()
            )
        return csv_dict

    def as_pyrocko_markers(self) -> list[marker.PhaseMarker]:
        """Convert the observed and modeled arrivals to a list of Pyrocko PhaseMarkers.

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
        """Convert the phase arrivals to Pyrocko markers.

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
        """Get the time window for phase arrivals.

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

    @computed_field
    @cached_property
    def n_receivers(self) -> int:
        """Number of receivers in the receiver set."""
        return len(self.receivers)

    def n_observations(self, phase: PhaseDescription) -> int:
        """Number of observations for a given phase."""
        n_observations = 0
        for receiver in self:
            if (arrival := receiver.phase_arrivals.get(phase)) and arrival.observed:
                n_observations += 1
        return n_observations

    async def get_waveforms(
        self,
        squirrel: Squirrel,
        seconds_before: float = 3.0,
        seconds_after: float = 5.0,
        phase: PhaseDescription | None = None,
        receivers: Iterable[Receiver] | None = None,
    ) -> list[Trace]:
        """Retrieves and restitutes waveforms for a given squirrel.

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
        if not times:
            return []

        tmin = min(times).timestamp() - seconds_before
        tmax = max(times).timestamp() + seconds_after
        nslc_ids = [(*receiver.nsl, "*") for receiver in receivers]
        async with SQUIRREL_SEM:
            traces = await asyncio.to_thread(
                squirrel.get_waveforms,
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
        filter_clipped: bool = False,
        freqlimits: tuple[float, float, float, float] = (0.01, 0.1, 25.0, 35.0),
        receivers: Iterable[Receiver] | None = None,
    ) -> list[Trace]:
        """Retrieves and restitutes waveforms for a given squirrel.

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
            filter_clipped (bool, optional): Whether to filter clipped traces.
                Defaults to False.

        Returns:
            list[Trace]: The restituted waveforms.
        """
        traces = await self.get_waveforms(
            squirrel,
            phase=phase,
            seconds_after=seconds_after + seconds_fade,
            seconds_before=seconds_before + seconds_fade,
            receivers=receivers,
        )
        traces = filter_clipped_traces(traces) if filter_clipped else traces
        if not traces:
            return []

        tmin = min(tr.tmin for tr in traces)
        tmax = max(tr.tmax for tr in traces)
        responses = await asyncio.to_thread(
            squirrel.get_responses,
            tmin=tmin,
            tmax=tmax,
            codes=[tr.nslc_id for tr in traces],
        )

        def get_response(tr: Trace) -> Any:
            for response in responses:
                if response.codes[:4] == tr.nslc_id:
                    return response
            raise ValueError(f"cannot find response for {tr.nslc_id}")

        restituted_traces = []
        for tr in traces:
            try:
                response = get_response(tr)
            except ValueError:
                logger.debug("cannot find response for %s", ".".join(tr.nslc_id))
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
        """Get the receiver object based on given NSL tuple.

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
        """Add receivers to the receiver set.

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
        """Retrieves a receiver object by its NSL (network, station, location) tuple.

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
        """Get a list of Pyrocko phase markers from all receivers.

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
        """Set the rundir for the detection model.

        Args:
            rundir (Path): The path to the rundir.
        """
        cls._rundir = rundir
        cls._receiver_cache = ReceiverCache(rundir / FILENAME_RECEIVERS)

    @property
    def magnitude(self) -> EventMagnitude | None:
        """Returns the magnitude of the event.

        If there are no magnitudes available, returns None.
        """
        return self.magnitudes[0] if self.magnitudes else None

    async def update(self) -> None:
        """Update detection in database.

        Doing this often requires a lot of I/O.
        """
        await self.save(update=True)

    async def save(self, file: Path | None = None, update: bool = False) -> None:
        """Dump the detection data to a file.

        After the detection is dumped, the receivers are dumped to a separate file and
        the receivers cache is cleared.

        Args:
            file (Path|None): The file to dump the detection to.
                If None, the rundir is used. Defaults to None.
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
            async with UPDATE_LOCK:
                async with aiofiles.open(file, "a") as f:
                    await f.write(f"{json_data}\n")

                receiver_file = self._rundir / FILENAME_RECEIVERS
                async with aiofiles.open(receiver_file, "a") as f:
                    await asyncio.shield(
                        f.write(f"{self.receivers.model_dump_json()}\n")
                    )

            self._receivers = None  # Free the memory

    def set_index(self, index: int, force: bool = False) -> None:
        """Set the index of the detection.

        Args:
            index (int): The index to set.
            force (bool, optional): Whether to force the index to be set.
                Defaults to False.

        Returns:
            None
        """
        if not force and self._detection_idx is not None:
            raise ValueError("cannot set index twice")
        self._detection_idx = index

    def set_uncertainty(self, uncertainty: DetectionUncertainty) -> None:
        """Set detection uncertainty.

        Args:
            uncertainty (DetectionUncertainty): detection uncertainty
        """
        self.uncertainty = uncertainty

    def add_magnitude(self, magnitude: EventMagnitude) -> None:
        """Add magnitude to detection.

        Args:
            magnitude (EventMagnitudeType): magnitude
        """
        for mag in self.magnitudes.copy():
            if type(magnitude) is type(mag):
                self.magnitudes.remove(mag)
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
        """Retrieves the event receivers associated with the detection.

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

            try:
                line = self._receiver_cache.get_line(self._detection_idx)
                receivers = EventReceivers.model_validate_json(line)
            except IndexError:
                receivers = None

            if not receivers or receivers.event_uid != self.uid:
                logger.warning("event %s uid mismatch, using brute search", self.time)
                try:
                    idx, line = self._receiver_cache.find_uid(self.uid)
                    receivers = EventReceivers.model_validate_json(line)
                    self.set_index(idx, force=True)
                except KeyError:
                    raise ValueError(f"uid mismatch for event {self.time}") from None

            self._receivers = receivers
        else:
            raise ValueError("cannot fetch receivers without set rundir and index")
        return self._receivers

    def as_pyrocko_event(self) -> Event:
        """Get detection as Pyrocko event.

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
        """Get detection as CSV line.

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
            "semblance": self.semblance,
            "n_stations": self.n_stations,
        }
        if self.uncertainty:
            csv_line.update(
                {
                    "uncertainty_horizontal": self.uncertainty.horizontal,
                    "uncertainty_vertical": self.uncertainty.vertical,
                }
            )
        csv_line["WKT_geom"] = self.as_wkt()

        for magnitude in self.magnitudes:
            csv_line.update(magnitude.csv_row())
        return csv_line

    def get_pyrocko_markers(self) -> list[marker.EventMarker | marker.PhaseMarker]:
        """Get detections as Pyrocko markers.

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
        """Save detection's Pyrocko markers to file.

        Args:
            filename (Path): path to marker file
        """
        logger.debug("dumping detection's Pyrocko markers to %s", filename)
        marker.save_markers(self.get_pyrocko_markers(), str(filename))

    def jitter_location(self, meters: float) -> Self:
        """Randomize detection location.

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

    def snuffle(
        self,
        squirrel: Squirrel,
        restituted: bool | MeasurementUnit = False,
    ) -> None:
        """Open snuffler for detection.

        Args:
            squirrel (Squirrel): The squirrel, holding the data
            restituted (bool, optional): Restitude the data. Defaults to False.
        """
        from pyrocko.trace import snuffle

        restitute_unit = "velocity" if restituted is True else restituted
        traces = (
            self.receivers.get_waveforms(squirrel)
            if not restitute_unit
            else self.receivers.get_waveforms_restituted(
                squirrel, quantity=restitute_unit
            )
        )
        snuffle(
            traces,
            markers=self.get_pyrocko_markers(),
            stations=[recv.as_pyrocko_station() for recv in self.receivers],
        )

    def __str__(self) -> str:
        # TODO: Add more information
        return str(self.time)
