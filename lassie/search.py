from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Literal

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, PrivateAttr
from pyrocko import parstack

from lassie.features import (
    FeatureExtractors,
    GroundMotionExtractor,
    LocalMagnitudeExtractor,
)
from lassie.images import ImageFunctions, WaveformImages
from lassie.models import Stations
from lassie.models.detection import EventDetection, EventDetections, PhaseDetection
from lassie.models.semblance import Semblance, SemblanceStats
from lassie.octree import NodeSplitError, Octree
from lassie.signals import Signal
from lassie.station_corrections import StationCorrections
from lassie.tracers import (
    CakeTracer,
    ConstantVelocityTracer,
    FastMarchingTracer,
    RayTracers,
)
from lassie.utils import PhaseDescription, alog_call, datetime_now, time_to_path
from lassie.waveforms import PyrockoSquirrel, WaveformProviderType

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from typing_extensions import Self

    from lassie.images.base import WaveformImage
    from lassie.octree import Node
    from lassie.tracers.base import RayTracer

logger = logging.getLogger(__name__)

SamplingRate = Literal[10, 20, 25, 50, 100]


class SearchProgress(BaseModel):
    time_progress: datetime | None = None
    semblance_stats: SemblanceStats = SemblanceStats()


class Search(BaseModel):
    project_dir: Path = Path(".")
    stations: Stations = Stations()
    data_provider: WaveformProviderType = PyrockoSquirrel()

    octree: Octree = Octree()
    image_functions: ImageFunctions = ImageFunctions()
    ray_tracers: RayTracers = RayTracers(
        root=[ConstantVelocityTracer(), CakeTracer(), FastMarchingTracer()]
    )
    station_corrections: StationCorrections = StationCorrections()
    event_features: list[FeatureExtractors] = [
        GroundMotionExtractor(),
        LocalMagnitudeExtractor(),
    ]

    sampling_rate: SamplingRate = 50
    detection_threshold: PositiveFloat = 0.05
    detection_blinding: timedelta = timedelta(seconds=2.0)

    image_mean_p: float = Field(default=1, ge=1.0, le=2.0)

    node_split_threshold: float = Field(default=0.9, gt=0.0, lt=1.0)
    window_length: timedelta = timedelta(minutes=5)

    n_threads_parstack: int = Field(default=0, ge=0)
    n_threads_argmax: PositiveInt = 4

    plot_octree_surface: bool = False
    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _progress: SearchProgress = PrivateAttr(SearchProgress())

    _shift_range: timedelta = PrivateAttr(timedelta(seconds=0.0))
    _window_padding: timedelta = PrivateAttr(timedelta(seconds=0.0))
    _distance_range: tuple[float, float] = PrivateAttr((0.0, 0.0))
    _travel_time_ranges: dict[
        PhaseDescription, tuple[timedelta, timedelta]
    ] = PrivateAttr({})

    _detections: EventDetections = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    # Signals
    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())
    _batch_processing_durations: Deque[timedelta] = PrivateAttr(
        default_factory=lambda: deque(maxlen=25)
    )

    def init_rundir(self, force=False) -> None:
        rundir = (
            self.project_dir / self._config_stem or f"run-{time_to_path(self.created)}"
        )
        self._rundir = rundir

        if rundir.exists() and not force:
            raise FileExistsError(f"Rundir {rundir} already exists")

        if rundir.exists() and force:
            create_time = time_to_path(
                datetime.fromtimestamp(rundir.stat().st_ctime)  # noqa
            )
            rundir_backup = rundir.with_name(f"{rundir.name}.bak-{create_time}")
            rundir.rename(rundir_backup)
            logger.info("created backup of existing rundir to %s", rundir_backup)

        if not rundir.exists():
            rundir.mkdir()

        file_logger = logging.FileHandler(self._rundir / "lassie.log")
        logging.root.addHandler(file_logger)
        self.write_config()

        logger.info("created new rundir %s", rundir)
        self._detections = EventDetections(rundir=rundir)

    def write_config(self, path: Path | None = None) -> None:
        rundir = self._rundir
        path = path or rundir / "search.json"

        logger.debug("writing search config to %s", path)
        path.write_text(self.model_dump_json(indent=2, exclude_unset=True))

        logger.debug("dumping stations...")
        self.stations.dump_pyrocko_stations(rundir / "pyrocko_stations.yaml")

        csv_dir = rundir / "csv"
        csv_dir.mkdir(exist_ok=True)
        self.stations.dump_csv(csv_dir / "stations.csv")

    @property
    def semblance_stats(self) -> SemblanceStats:
        return self._progress.semblance_stats

    def set_progress(self, time: datetime) -> None:
        self._progress.time_progress = time
        progress_file = self._rundir / "progress.json"
        progress_file.write_text(self._progress.model_dump_json())

    def init_boundaries(self) -> None:
        """Initialise search."""
        # Grid/receiver distances
        distances = self.octree.distances_stations(self.stations)
        self._distance_range = (distances.min(), distances.max())

        # Timing ranges
        for phase, tracer in self.ray_tracers.iter_phase_tracer(
            phases=self.image_functions.get_phases()
        ):
            traveltimes = tracer.get_travel_times(phase, self.octree, self.stations)
            self._travel_time_ranges[phase] = (
                timedelta(seconds=np.nanmin(traveltimes)),
                timedelta(seconds=np.nanmax(traveltimes)),
            )
            logger.info(
                "time shift ranges: %s / %s - %s",
                phase,
                *self._travel_time_ranges[phase],
            )

        # TODO: minimum shift is calculated on the coarse octree grid, which is
        # not necessarily the same as the fine grid used for semblance calculation
        shift_min = min(chain.from_iterable(self._travel_time_ranges.values()))
        shift_max = max(chain.from_iterable(self._travel_time_ranges.values()))
        self._shift_range = shift_max - shift_min

        self._window_padding = (
            self._shift_range
            + self.detection_blinding
            + self.image_functions.get_blinding()
        )
        if self.window_length < 2 * self._window_padding + self._shift_range:
            raise ValueError(
                f"window length {self.window_length} is too short for the "
                f"theoretical travel time range {self._shift_range} and "
                f"cummulative window padding of {self._window_padding}."
                " Increase the window_length time."
            )

        logger.info("using trace window padding: %s", self._window_padding)
        logger.info("time shift range %s", self._shift_range)
        logger.info(
            "source-station distance range: %.1f - %.1f m",
            *self._distance_range,
        )

    async def prepare(self) -> None:
        logger.info("preparing search...")
        self.data_provider.prepare(self.stations)
        await self.ray_tracers.prepare(
            self.octree,
            self.stations,
            phases=self.image_functions.get_phases(),
        )
        self.init_boundaries()

    async def start(self, force_rundir: bool = False) -> None:
        await self.prepare()
        self.init_rundir(force_rundir)
        logger.info("starting search...")
        batch_processing_start = datetime_now()
        processing_start = datetime_now()

        if self._progress.time_progress:
            logger.info("continuing search from %s", self._progress.time_progress)

        async for batch in self.data_provider.iter_batches(
            window_increment=self.window_length,
            window_padding=self._window_padding,
            start_time=self._progress.time_progress,
        ):
            batch.clean_traces()

            if batch.is_empty():
                logger.warning("empty batch %s", batch.log_str())
                continue

            if batch.duration < 2 * self._window_padding:
                logger.warning(
                    "duration of batch %s too short %s",
                    batch.log_str(),
                    batch.duration,
                )
                continue

            search_block = SearchTraces(
                parent=self,
                traces=batch.traces,
                start_time=batch.start_time,
                end_time=batch.end_time,
            )
            detections, semblance_trace = await search_block.search()

            self._detections.add_semblance(semblance_trace)
            for detection in detections:
                if detection.in_bounds:
                    await self.add_features(detection)

                self._detections.add(detection)
                await self._new_detection.emit(detection)

            if self._detections.n_detections % 50 == 0:
                self._detections.dump_detections(jitter_location=self.octree.size_limit)

            processing_time = datetime_now() - batch_processing_start
            self._batch_processing_durations.append(processing_time)
            if batch.n_batches:
                percent_processed = ((batch.i_batch + 1) / batch.n_batches) * 100
            else:
                percent_processed = 0.0
            logger.info(
                "%s%% processed - batch %s in (%s/s)",
                f"{percent_processed:.1f}" if percent_processed else "??",
                batch.log_str(),
                batch.cumulative_duration / processing_time.total_seconds(),
            )
            if batch.n_batches:
                remaining_time = (
                    sum(self._batch_processing_durations, timedelta())
                    / len(self._batch_processing_durations)
                    * (batch.n_batches - batch.i_batch - 1)
                )
                logger.info(
                    "%s remaining - estimated finish at %s",
                    remaining_time,
                    datetime.now() + remaining_time,  # noqa: DTZ005
                )

            batch_processing_start = datetime_now()
            self.set_progress(batch.end_time)

        self._detections.dump_detections(jitter_location=self.octree.size_limit)
        logger.info("finished search in %s", datetime_now() - processing_start)
        logger.info("found %d detections", self._detections.n_detections)

    async def add_features(self, event: EventDetection) -> None:
        try:
            squirrel = self.data_provider.get_squirrel()
        except NotImplementedError:
            return

        for extractor in self.event_features:
            logger.info("adding features from %s", extractor.feature)
            await extractor.add_features(squirrel, event)

    @classmethod
    def load_rundir(cls, rundir: Path) -> Self:
        search_file = rundir / "search.json"
        search = cls.model_validate_json(search_file.read_bytes())
        search._rundir = rundir
        search._detections = EventDetections.load_rundir(rundir)

        progress_file = rundir / "progress.json"
        if progress_file.exists():
            search._progress = SearchProgress.model_validate_json(
                progress_file.read_text()
            )
        return search

    @classmethod
    def from_config(
        cls,
        filename: Path,
    ) -> Self:
        model = super().model_validate_json(filename.read_text())
        # Make relative paths absolute
        filename = Path(filename)
        base_dir = filename.absolute().parent
        for name in model.model_fields_set:
            value = getattr(model, name)
            if isinstance(value, Path) and not value.absolute():
                setattr(model, name, value.relative_to(base_dir))
        model._config_stem = filename.stem
        return model

    def __del__(self) -> None:
        # FIXME: Replace with signal overserver?
        if hasattr(self, "_detections"):
            with contextlib.suppress(Exception):
                self._detections.dump_detections(jitter_location=self.octree.size_limit)


class SearchTraces:
    _images: dict[float | None, WaveformImages]

    def __init__(
        self,
        parent: Search,
        traces: list[Trace],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.parent = parent
        self.traces = traces
        self.start_time = start_time
        self.end_time = end_time

        self._images = {}

    def _n_samples_semblance(self) -> int:
        """Number of samples to use for semblance calculation, includes padding."""
        parent = self.parent
        window_padding = parent._window_padding
        time_span = (self.end_time + window_padding) - (
            self.start_time - window_padding
        )
        return int(round(time_span.total_seconds() * parent.sampling_rate))

    @alog_call
    async def calculate_semblance(
        self,
        octree: Octree,
        image: WaveformImage,
        ray_tracer: RayTracer,
        n_samples_semblance: int,
        semblance_data: np.ndarray,
    ) -> np.ndarray:
        logger.debug("stacking image %s", image.image_function.name)
        parent = self.parent

        traveltimes = ray_tracer.get_travel_times(image.phase, octree, image.stations)

        if parent.station_corrections:
            station_delays = parent.station_corrections.get_delays(
                image.stations.get_all_nsl(), image.phase
            )
            traveltimes += station_delays[np.newaxis, :]

        traveltimes_bad = np.isnan(traveltimes)
        traveltimes[traveltimes_bad] = 0.0
        station_contribution = (~traveltimes_bad).sum(axis=1, dtype=np.float32)

        shifts = np.round(-traveltimes / image.delta_t).astype(np.int32)
        weights = np.full_like(shifts, fill_value=image.weight, dtype=np.float32)

        # Normalize by number of station contribution
        with np.errstate(divide="ignore", invalid="ignore"):
            weights /= station_contribution[:, np.newaxis]
        weights[traveltimes_bad] = 0.0

        semblance_data, offsets = await asyncio.to_thread(
            parstack.parstack,
            arrays=image.get_trace_data(),
            offsets=image.get_offsets(self.start_time - parent._window_padding),
            shifts=shifts,
            weights=weights,
            lengthout=n_samples_semblance,
            result=semblance_data,
            dtype=np.float32,
            method=0,
            nparallel=parent.n_threads_parstack,
        )
        return semblance_data

    async def get_images(self, sampling_rate: float | None = None) -> WaveformImages:
        """
        Retrieves waveform images for the specified sampling rate.

        Args:
            sampling_rate (float | None, optional): The desired sampling rate in Hz.
                Defaults to None.

        Returns:
            WaveformImages: The waveform images for the specified sampling rate.
        """
        if None not in self._images:
            images = await self.parent.image_functions.process_traces(self.traces)
            images.set_stations(self.parent.stations)
            images.apply_exponent(self.parent.image_mean_p)
            self._images[None] = images

        if sampling_rate not in self._images:
            if not isinstance(sampling_rate, float):
                raise TypeError("sampling rate has to be a float or int")
            images_resampled = deepcopy(self._images[None])

            logger.debug("downsampling images to %g Hz", sampling_rate)
            images_resampled.downsample(sampling_rate, max_normalize=True)
            self._images[sampling_rate] = images_resampled

        return self._images[sampling_rate]

    async def search(
        self,
        octree: Octree | None = None,
    ) -> tuple[list[EventDetection], Trace]:
        """Searches for events in the given traces.

        Args:
            octree (Octree | None, optional): The octree to use for the search.
                Defaults to None.

        Returns:
            tuple[list[EventDetection], Trace]: The event detections and the
                semblance traces used for the search.
        """
        parent = self.parent
        sampling_rate = parent.sampling_rate

        octree = octree or parent.octree.copy(deep=True)
        images = await self.get_images(sampling_rate=float(sampling_rate))

        padding_samples = int(
            round(parent._window_padding.total_seconds() * sampling_rate)
        )
        semblance = Semblance(
            n_nodes=octree.n_nodes,
            n_samples=self._n_samples_semblance(),
            start_time=self.start_time,
            sampling_rate=sampling_rate,
            padding_samples=padding_samples,
        )

        for image in images:
            await self.calculate_semblance(
                octree=octree,
                image=image,
                ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                semblance_data=semblance.semblance_unpadded,
                n_samples_semblance=semblance.n_samples_unpadded,
            )
        semblance.apply_exponent(1.0 / parent.image_mean_p)
        semblance.normalize(images.cumulative_weight())

        parent.semblance_stats.update(semblance.get_stats())
        logger.debug("semblance stats: %s", parent.semblance_stats)

        detection_idx, detection_semblance = semblance.find_peaks(
            height=parent.detection_threshold**parent.image_mean_p,
            prominence=parent.detection_threshold**parent.image_mean_p,
            distance=round(parent.detection_blinding.total_seconds() * sampling_rate),
        )

        if detection_idx.size == 0:
            return [], semblance.get_trace()

        # Split Octree nodes above a semblance threshold. Once octree for all detections
        # in frame
        maxima_node_idx = await semblance.maxima_node_idx()
        refine_nodes: set[Node] = set()
        for time_idx, semblance_detection in zip(
            detection_idx, detection_semblance, strict=True
        ):
            octree.map_semblance(semblance.semblance[:, time_idx])
            node_idx = maxima_node_idx[time_idx]
            source_node = octree[node_idx]

            if not source_node.can_split():
                continue

            split_nodes = octree.get_nodes(
                semblance_detection * parent.node_split_threshold
            )
            refine_nodes.update(split_nodes)

        # refine_nodes is empty when all sources fall into smallest octree nodes
        if refine_nodes:
            logger.info("energy detected, refining %d nodes", len(refine_nodes))
            for node in refine_nodes:
                try:
                    node.split()
                except NodeSplitError:
                    continue

            del semblance
            return await self.search(octree)

        detections = []
        for time_idx, semblance_detection in zip(
            detection_idx, detection_semblance, strict=True
        ):
            time = self.start_time + timedelta(seconds=time_idx / sampling_rate)
            octree.map_semblance(semblance.semblance[:, time_idx])

            node_idx = (await semblance.maxima_node_idx())[time_idx]
            source_node = octree[node_idx]
            source_location = source_node.as_location()

            detection = EventDetection(
                time=time,
                semblance=float(semblance_detection),
                distance_border=source_node.distance_border,
                in_bounds=octree.is_node_in_bounds(source_node),
                **source_location.model_dump(),
            )

            # Attach modelled and picked arrivals to receivers
            for image in await self.get_images(sampling_rate=None):
                ray_tracer = parent.ray_tracers.get_phase_tracer(image.phase)
                arrivals_model = ray_tracer.get_arrivals(
                    phase=image.phase,
                    event_time=time,
                    source=source_location,
                    receivers=image.stations.stations,
                )
                arrivals_observed = image.search_phase_arrivals(
                    modelled_arrivals=[
                        arr.time if arr else None for arr in arrivals_model
                    ]
                )

                phase_detections = [
                    PhaseDetection(phase=image.phase, model=mod, observed=obs)
                    if mod
                    else None
                    for mod, obs in zip(arrivals_model, arrivals_observed, strict=True)
                ]
                detection.receivers.add_receivers(
                    stations=image.stations.stations,
                    phase_arrivals=phase_detections,
                )

            detections.append(detection)
            # detection.plot()

        # plot_octree_movie(octree, semblance, file=Path("/tmp/test.mp4"))
        if parent.plot_octree_surface:
            octree.map_semblance(semblance.maximum_node_semblance())
            parent._plot_octree_surface(
                octree, time=self.start_time, detections=detections
            )

        return detections, semblance.get_trace()
