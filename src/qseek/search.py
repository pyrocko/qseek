from __future__ import annotations

import asyncio
import logging
import shutil
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Literal, Self, Sequence

import numpy as np
import psutil
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    computed_field,
    field_validator,
)
from scipy import stats

from qseek.cache_lru import CACHES
from qseek.corrections.corrections import StationCorrectionType, corrections_from_path
from qseek.features import FeatureExtractorType
from qseek.images.images import ImageFunctions, WaveformImages
from qseek.magnitudes import EventMagnitudeCalculatorType
from qseek.models import StationInventory
from qseek.models.catalog import EventCatalog
from qseek.models.detection import EventDetection, PhaseDetection
from qseek.models.detection_uncertainty import DetectionUncertainty
from qseek.models.station import Station
from qseek.octree import Octree
from qseek.pre_processing.frequency_filters import Bandpass
from qseek.pre_processing.module import Downsample, PreProcessing
from qseek.reduce import DelaySumReduce
from qseek.signals import Signal
from qseek.station_weights import StationWeights
from qseek.stats import RuntimeStats, Stats
from qseek.tracers.tracers import RayTracer, RayTracers
from qseek.utils import (
    BackgroundTasks,
    CpuCount,
    PhaseDescription,
    _get_cpu_count,
    datetime_now,
    get_total_memory,
    human_readable_bytes,
    time_to_path,
)
from qseek.waveforms.base import WaveformBatch
from qseek.waveforms.providers import PyrockoSquirrel, WaveformProviderType

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table

    from qseek.octree import Node


logger = logging.getLogger(__name__)

SamplingRate = Literal[10, 20, 25, 50, 100, 200, 400]
IgnoreBoundary = Literal[False, "with_surface", "without_surface"]

KM = 1e3


class SearchStats(Stats):
    project_name: str = "qseek"
    batch_time: datetime = datetime.min.replace(tzinfo=timezone.utc)
    batch_count: int = 0
    batch_count_total: int = 0
    processed_time: timedelta = timedelta(seconds=0.0)
    processed_bytes: int = 0
    processing_time: timedelta = timedelta(seconds=0.0)
    current_processing_rate: float = 0.0
    current_processing_speed: timedelta = timedelta(seconds=0.0)

    current_stations: int = 0
    total_stations: int = 0

    current_networks: int = 0
    total_networks: int = 0

    memory_total: ByteSize = Field(default_factory=lambda: ByteSize(get_total_memory()))

    _search_start: datetime = PrivateAttr(default_factory=datetime_now)
    _batch_processing_times: Deque[timedelta] = PrivateAttr(
        default_factory=lambda: deque(maxlen=25)
    )
    _position: int = PrivateAttr(0)
    _process: psutil.Process = PrivateAttr(default_factory=psutil.Process)

    @computed_field
    @property
    def time_remaining(self) -> timedelta:
        if not self.batch_count or not self.batch_count_total:
            return timedelta()

        remaining_batches = self.batch_count_total - self.batch_count
        if not remaining_batches:
            return timedelta()

        elapsed_time = datetime_now() - self._search_start
        return (elapsed_time / self.batch_count) * remaining_batches

    @computed_field
    @property
    def processing_rate(self) -> float:
        """Calculate the processing rate of the search.

        Returns:
            float: The processing rate in bytes per second.
        """
        if not self.processing_time:
            return 0.0
        return self.processed_bytes / self.processing_time.total_seconds()

    @computed_field
    @property
    def processing_speed(self) -> timedelta:
        if not self.processing_time:
            return timedelta(seconds=0.0)
        return self.processed_time / self.processing_time.total_seconds()

    @computed_field
    @property
    def processed_percent(self) -> float:
        """Calculate the percentage of processed batches.

        Returns:
            float: The percentage of processed batches.
        """
        if not self.batch_count_total:
            return 0.0
        return self.batch_count / self.batch_count_total * 100.0

    @computed_field
    @property
    def memory_used(self) -> int:
        return self._process.memory_info().rss

    @computed_field
    @property
    def cpu_percent(self) -> float:
        return self._process.cpu_percent(interval=None)

    def reset_search_begin(self) -> None:
        self._search_start = datetime_now()

    def add_processed_batch(
        self,
        batch: WaveformBatch,
        duration: timedelta,
        show_log: bool = False,
    ) -> None:
        self.batch_count = batch.i_batch + 1
        self.batch_count_total = batch.n_batches
        self.batch_time = batch.end_time
        self.processed_bytes += batch.nbytes
        self.processed_time += batch.duration
        self.processing_time += duration
        self.current_processing_rate = batch.nbytes / duration.total_seconds()
        self.current_processing_speed = batch.duration / duration.total_seconds()
        self.current_stations = batch.n_stations
        self.current_networks = batch.n_networks

        self._batch_processing_times.append(duration)
        if show_log:
            self.log()

    def log(self) -> None:
        log_str = (
            f"{self.batch_count + 1}/{self.batch_count_total or '?'} "
            f"{str(self.batch_time)[:22]}"
        )
        logger.info(
            "%s%% processed - batch %s in %s",
            f"{self.processed_percent:.1f}" if self.processed_percent else "??",
            log_str,
            self._batch_processing_times[-1],
        )
        logger.info(
            "processing rate %s/s", human_readable_bytes(self.current_processing_rate)
        )

    def _populate_table(self, table: Table) -> None:
        def tts(duration: timedelta) -> str:
            return str(duration).split(".")[0]

        table.add_row(
            "Project",
            f"[bold]{self.project_name}[/bold]",
        )
        table.add_row(
            "Current Stations",
            f"{self.current_stations}/{self.total_stations}"
            f" ({self.current_networks}/{self.total_networks} networks)",
        )
        table.add_row(
            "Progress ",
            f"[bold]{self.processed_percent:.1f}%[/bold]"
            f" ([bold]{self.batch_count}[/bold]/{self.batch_count_total or '?'},"
            f" {self.batch_time.strftime('%Y-%m-%d %H:%M:%S')})",
        )
        table.add_row(
            "Remaining Time",
            f"[bold]{tts(self.time_remaining)}[/bold], "
            f"finish at {datetime.now() + self.time_remaining:%c}",  # noqa: DTZ005
        )
        table.add_row(
            "Processing rate",
            f"[bold]{human_readable_bytes(self.processing_rate)}/s[/bold]"
            f" ({tts(self.processing_speed)} m/s)",
        )
        table.add_row(
            "Resources",
            f"CPU {self.cpu_percent:>6.1f}%, "
            f"RAM {human_readable_bytes(self.memory_used, decimal=True)}"
            f"/{self.memory_total.human_readable(decimal=True)}",
        )
        table.add_row(
            "Caches",
            f"{human_readable_bytes(CACHES.get_fill_bytes())}"
            f" {CACHES.get_fill_level() * 100:.1f}% - "
            f"{' '.join(cache.__rich__() for cache in CACHES)}",
        )


class SearchProgress(BaseModel):
    time_progress: datetime | None = None


class Search(BaseModel):
    project_dir: Path = Path(".")
    stations: StationInventory = Field(
        default=StationInventory(),
        description="Station inventory from StationXML or Pyrocko Station YAML.",
    )
    data_provider: WaveformProviderType = Field(
        default_factory=PyrockoSquirrel.model_construct,
        description="Data provider for waveform data.",
    )
    pre_processing: PreProcessing = Field(
        default_factory=lambda: PreProcessing(root=[Downsample(), Bandpass()]),
        description="Pre-processing steps for waveform data.",
    )

    octree: Octree = Field(
        default_factory=Octree,
        description="Octree volume for the search.",
    )

    image_functions: ImageFunctions = Field(
        default_factory=ImageFunctions,
        description="Image functions for waveform processing and "
        "phase on-set detection.",
    )
    ray_tracers: RayTracers = Field(
        default_factory=lambda: RayTracers(
            root=[tracer() for tracer in RayTracer.get_subclasses()]
        ),
        description="List of ray tracers for travel time calculation.",
    )
    station_weights: StationWeights | None = Field(
        default_factory=StationWeights,
        description="Station weighting based on station density and "
        "source-station distance.",
    )
    station_corrections: StationCorrectionType | None = Field(
        default=None,
        description="Apply station corrections extracted from a previous run or a path"
        " to a directory with station correction files.",
    )
    magnitudes: list[EventMagnitudeCalculatorType] = Field(
        default_factory=list,
        description="Magnitude calculators to use.",
    )
    features: list[FeatureExtractorType] = Field(
        default_factory=list,
        description="Event features to extract.",
    )

    semblance_sampling_rate: SamplingRate = Field(
        default=100,
        description="Sampling rate for the semblance image function. "
        "Choose from `10, 20, 25, 50, 100, 200 or 400` Hz.",
    )
    detection_threshold: Literal["MAD"] | PositiveFloat = Field(
        default="MAD",
        description="Detection threshold for semblance.",
    )
    pick_confidence_threshold: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Confidence threshold for picking.",
    )
    min_stations: int = Field(
        default=3,
        ge=0,
        description="Minimum number of stations required for"
        " detection and localization.",
    )
    ignore_boundary: IgnoreBoundary = Field(
        default=False,
        description="Ignore events that are inside the first root node layer of"
        " the octree. If `with_surface`, all events inside the boundaries of the volume"
        " are absorbed. If `without_surface`, events at the surface are not absorbed.",
    )
    ignore_boundary_width: float | Literal["root_node_size"] = Field(
        default="root_node_size",
        description="Width of the absorbing boundary around the octree volume. "
        "If 'octree' the width is set to the root node size of the octree.",
    )
    node_interpolation: bool = Field(
        default=True,
        description="Interpolate intranode locations for detected events using radial"
        " basis functions. If `False`, the node center location is used for "
        "the event hypocentre.",
    )
    detection_blinding: timedelta = Field(
        default=timedelta(seconds=1.0),
        description="Blinding time in seconds before and after the detection peak. "
        "This is used to avoid detecting the same event multiple times. "
        "Default is 2 seconds.",
    )

    window_length: timedelta = Field(
        default=timedelta(minutes=5),
        description="Window length for processing. Smaller window size will be less RAM"
        " consuming. Default is 5 minutes.",
    )

    n_threads: CpuCount = Field(
        default="auto",
        description="Number of threads for stacking and migration. "
        "`'auto'` will use the maximum number of cores and leaves resources "
        "for I/O and other work. `0` uses all available cores.",
    )

    save_images: bool = Field(
        default=False,
        description="Save annotation images to disk for debugging and analysis.",
    )
    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _progress: SearchProgress = PrivateAttr(SearchProgress())

    _last_detection_export: int = 0

    _catalog: EventCatalog = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    _compute_semaphore: asyncio.Semaphore = PrivateAttr(
        asyncio.Semaphore(max(1, _get_cpu_count() - 4))
    )

    # Signals
    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())

    model_config = ConfigDict(extra="forbid")

    @field_validator("station_corrections", mode="before")
    @classmethod
    def load_path(cls, v: Any) -> StationCorrectionType:
        if isinstance(v, str):
            path = Path(v)
            if not path.is_dir():
                raise ValueError(f"station_corrections path {path} is not a directory")
            return corrections_from_path(path)
        return v

    def init_rundir(self, force: bool = False, create_backup: bool = True) -> None:
        rundir = (
            self.project_dir / self._config_stem or f"run-{time_to_path(self.created)}"
        )
        self._rundir = rundir

        if rundir.exists() and not force:
            raise FileExistsError(f"Rundir {rundir} already exists")

        if rundir.exists() and force and create_backup:
            create_time = time_to_path(
                datetime.fromtimestamp(rundir.stat().st_ctime)  # noqa
            ).split(".")[0]
            rundir_backup = rundir.with_name(f"{rundir.name}.bak-{create_time}")
            rundir.rename(rundir_backup)
            logger.info("backing up existing rundir to %s", rundir_backup)

        if rundir.exists() and not create_backup:
            logger.warning("overwriting existing rundir %s", rundir)
            # Move directory first for slow HPC storage
            rm_rundir = rundir.move(rundir.with_suffix("-del"))
            shutil.rmtree(rm_rundir)

        if not rundir.exists():
            logger.info("creating rundir %s", rundir)
            rundir.mkdir()

        self._init_logging()

        logger.info("created new rundir %s", rundir)
        self._catalog = EventCatalog(rundir=rundir)

    @property
    def catalog(self) -> EventCatalog:
        return self._catalog

    def _init_logging(self) -> None:
        file_logger = logging.FileHandler(self._rundir / "qseek.log")
        logging.root.addHandler(file_logger)

    def create_folders(self, path: Path | None = None) -> None:
        rundir = path or self._rundir
        csv_dir = rundir / "csv"
        csv_dir.mkdir(exist_ok=True)

    def write_config(self, path: Path | None = None) -> None:
        rundir = path or self._rundir
        path = rundir / "search.json"

        logger.debug("writing search config to %s", path)
        path.write_text(self.model_dump_json(indent=2))

        logger.debug("writing stations")
        self.stations.export_pyrocko_stations(rundir / "pyrocko_stations.yaml")

        csv_dir = rundir / "csv"
        csv_dir.mkdir(exist_ok=True)
        self.stations.export_csv(csv_dir / "stations.csv")

    def set_progress(self, time: datetime) -> None:
        self._progress.time_progress = time
        progress_file = self._rundir / "progress.json"
        progress_file.write_text(self._progress.model_dump_json())

    async def get_window_padding(self) -> timedelta:
        """Get window padding length based on maximum travel time shifts.

        This is a calculation based on the maximum travel time shifts from the ray
        tracers, the image function blinding, and the detection blinding.


        Returns:
            timedelta: Window padding length.
        """
        # TODO: minimum shift is calculated on the coarse octree grid, which is
        # not necessarily the same as the fine grid used for semblance calculation
        shift_min, shift_max = await self.ray_tracers.get_travel_time_span(
            self.octree,
            list(self.stations),
            self.image_functions.get_phases(),
        )
        shift_range = shift_max - shift_min
        logger.info("maximum travel time shift %s", shift_max)

        return (
            shift_range + self.image_functions.get_blinding() + self.detection_blinding
        )

    async def prepare(self) -> None:
        """Prepares the search by initializing necessary components and data.

        This method prepares the search by performing the following steps:
        1. Prepares the data provider with the given stations.
        2. Prepares the ray tracers with the octree, stations, phases, and rundir.
        3. Prepares each magnitude with the octree and stations.
        4. Prepares the station corrections with the stations, octree, and phases.
        5. Initializes the boundaries.

        Note: This method is asynchronous.

        Returns:
            None
        """
        logger.info("preparing search components")
        self.stations.prepare(self.octree.location)

        self.data_provider.prepare(self.stations)
        self.stations.filter_stations(self.data_provider.available_nsls())

        distances = self.octree.distances_stations(self.stations)
        logger.info(
            "source-station distance range: %.2f - %.2f km",
            distances.min() / KM,
            distances.max() / KM,
        )

        await self.ray_tracers.prepare(
            self.octree,
            self.stations,
            phases=self.image_functions.get_phases(),
            rundir=self._rundir,
        )
        await self.pre_processing.prepare()
        await self.image_functions.prepare()

        if self.station_weights:
            self.station_weights.prepare(self.stations, self.octree)

        if self.station_corrections:
            await self.station_corrections.prepare(
                self.stations,
                self.octree,
                self.image_functions.get_phases(),
                self._rundir,
            )
        for magnitude in self.magnitudes:
            await magnitude.prepare(self.octree, self.stations)

        self._catalog.prepare()

    async def start(
        self,
        force_rundir: bool = False,
        create_backup: bool = True,
    ) -> None:
        if not self.has_rundir():
            self.init_rundir(force=force_rundir, create_backup=create_backup)

        self.create_folders()
        await self.prepare()
        self.write_config()

        if self._progress.time_progress:
            logger.info("continuing search from %s", self._progress.time_progress)
            await self._catalog.check(repair=True)
            await self._catalog.filter_events_by_time(
                start_time=None,
                end_time=self._progress.time_progress,
            )
        else:
            logger.info("starting search")

        window_padding = await self.get_window_padding()
        logger.info("window padding length is: %s", window_padding)

        batches = self.data_provider.iter_batches(
            window_increment=self.window_length,
            window_padding=window_padding,
            start_time=self._progress.time_progress,
            min_length=2 * window_padding,
            min_stations=self.min_stations,
        )
        pre_processed_batches = self.pre_processing.iter_batches(batches)

        stats = SearchStats(
            project_name=self._rundir.name,
            total_stations=self.stations.n_stations,
            total_networks=self.stations.n_networks,
        )
        stats.reset_search_begin()

        processing_start = datetime_now()
        console = asyncio.create_task(RuntimeStats.live_view())

        search_octree = OctreeSearch(
            ray_tracers=self.ray_tracers,
            window_padding=window_padding,
            station_corrections=self.station_corrections,
            distance_weights=self.station_weights,
            detection_threshold=self.detection_threshold,
            pick_confidence_threshold=self.pick_confidence_threshold,
            node_interpolation=self.node_interpolation,
            ignore_boundary=self.ignore_boundary,
            ignore_boundary_width=self.ignore_boundary_width,
        )

        async for images, batch in self.image_functions.iter_images(
            pre_processed_batches
        ):
            batch_processing_start = datetime_now()

            images.set_stations(self.stations)
            images.resample(self.semblance_sampling_rate)

            detections, semblance_trace = await search_octree.search(
                images=images,
                octree=self.octree.reset(),
                n_threads=self.n_threads,
            )

            await self._catalog.save_semblance_trace(semblance_trace)
            if detections:
                BackgroundTasks.create_task(self.new_detections(detections))
            if self.save_images:
                await images.save_mseed(self._rundir / "images")

            stats.add_processed_batch(
                batch,
                duration=datetime_now() - batch_processing_start,
                show_log=True,
            )
            self.set_progress(batch.end_time)

        await BackgroundTasks.wait_all()
        await self._catalog.save()
        await self._catalog.export_detections(
            jitter_location=self.octree.smallest_node_size()
        )
        console.cancel()
        logger.info("finished search in %s", datetime_now() - processing_start)
        logger.info("detected %d events", self._catalog.n_events)

    async def new_detections(self, detections: list[EventDetection]) -> None:
        """Process new detections.

        Args:
            detections (list[EventDetection]): List of new event detections.
        """
        catalog = self.catalog
        await asyncio.gather(
            *(self.add_magnitude_and_features(det) for det in detections)
        )

        for detection in detections:
            await catalog.add(
                detection,
                jitter_location=self.octree.smallest_node_size(),
            )
            await self._new_detection.emit(detection)

        if not catalog.n_events:
            return

    async def add_magnitude_and_features(
        self,
        event: EventDetection,
        recalculate: bool = True,
    ) -> EventDetection:
        """Adds magnitude and features to the given event.

        Args:
            event (EventDetection): The event to add magnitude and features to.
            recalculate (bool, optional): Whether to overwrite existing magnitudes and
                features. Defaults to True.
        """
        if not event.in_bounds:
            return event

        async with self._compute_semaphore:
            for mag_calculator in self.magnitudes:
                if not recalculate and mag_calculator.has_magnitude(event):
                    continue
                logger.debug("adding magnitude from %s", mag_calculator.magnitude)
                await mag_calculator.add_magnitude(self.data_provider, event)

            for feature_calculator in self.features:
                logger.debug("adding features from %s", feature_calculator.feature)
                await feature_calculator.add_features(self.data_provider, event)
        return event

    @classmethod
    def load_rundir(cls, rundir: Path) -> Self:
        search_file = rundir / "search.json"
        search = cls.model_validate_json(search_file.read_bytes())
        search._rundir = rundir
        search._catalog = EventCatalog.load_rundir(rundir)

        progress_file = rundir / "progress.json"
        if progress_file.exists():
            search._progress = SearchProgress.model_validate_json(
                progress_file.read_text()
            )

        search._init_logging()
        return search

    @classmethod
    def from_config(
        cls,
        filename: Path,
    ) -> Self:
        if filename.is_dir():
            filename = filename / "search.json"
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

    def has_rundir(self) -> bool:
        return hasattr(self, "_rundir") and self._rundir.exists()

    # def __del__(self) -> None:
    # FIXME: Replace with signal overserver?
    # if hasattr(self, "_detections"):
    # with contextlib.suppress(Exception):
    #     asyncio.run(
    #         self._detections.export_detections(
    #             jitter_location=self.octree.smallest_node_size()
    #         )
    #     )


class OctreeSearch:
    def __init__(
        self,
        ray_tracers: RayTracers,
        window_padding: timedelta,
        station_corrections: StationCorrectionType | None = None,
        distance_weights: StationWeights | None = None,
        detection_threshold: float | Literal["MAD"] = "MAD",
        detection_blinding: timedelta = timedelta(seconds=1.0),
        pick_confidence_threshold: float = 0.3,
        ignore_boundary: IgnoreBoundary = "with_surface",
        ignore_boundary_width: float | Literal["root_node_size"] = "root_node_size",
        node_interpolation: bool = True,
        attach_arrivals: bool = True,
        neighbor_search: bool = True,
        semblance_density_search: bool = True,
    ) -> None:
        """Initializes the OctreeSearch object.

        Args:
            ray_tracers (RayTracers): The ray tracers to use for travel time
                calculations.
            window_padding (timedelta): The padding time for the search window.
            station_corrections (StationCorrectionType | None, optional): The
                station corrections to apply. Defaults to None.
            distance_weights (DistanceWeights | None, optional): The distance
                weights to apply. Defaults to None.
            detection_threshold (float | Literal["MAD"], optional): The detection
                threshold for the search. If "MAD", the threshold is set to 10 times
                the median absolute deviation of the semblance. Defaults to "MAD".
                picking. Defaults to 0.3.
            detection_blinding (timedelta, optional): The blinding time for the
                detection. Defaults to 1 second.
            pick_confidence_threshold (float, optional): The confidence threshold for
                picking. Defaults to 0.3.
            ignore_boundary
                    (Literal[False, "with_surface", "without_surface"], optional):
                Whether to ignore events at the boundary of the octree.
                If "with_surface", all events inside the boundaries of the volume are
                irgnored. If "without_surface", events at the surface are not ignored.
                Defaults to False.
            ignore_boundary_width (float | Literal["root_node_size"], optional): The
                width of the absorbing boundary around the octree volume. If
                'octree' the width is set to the root node size of the octree.
                Defaults to "root_node_size".
            node_interpolation (bool, optional): Whether to interpolate the event
                location within the octree node. Defaults to True.
            attach_arrivals (bool, optional): Whether to attach phase arrivals to the
                detection. Defaults to True.
            neighbor_search (bool, optional): Whether to search for phase arrivals at
                neighboring stations. Defaults to True.
            semblance_density_search (bool, optional): Whether to perform a density
                based search for the semblance peak. If False, the maximum value is
                used. Defaults to True.
        """
        self.ray_tracers = ray_tracers

        self.window_padding = window_padding

        self.detection_threshold = detection_threshold
        self.blinding = detection_blinding
        self.pick_confidence_threshold = pick_confidence_threshold
        self.node_interpolation = node_interpolation
        self.attach_arrivals = attach_arrivals
        self.neighbor_search = neighbor_search
        self.semblance_density_search = semblance_density_search

        self.ignore_boundary = ignore_boundary
        self.ignore_boundary_width = ignore_boundary_width

        self.station_corrections = station_corrections
        self.distance_weights = distance_weights

    def set_ray_tracers(self, ray_tracers: RayTracers) -> None:
        self.ray_tracers = ray_tracers

    async def get_traveltimes(
        self,
        stations: Sequence[Station],
        nodes: Sequence[Node],
        phase: PhaseDescription,
    ) -> np.ndarray:
        ray_tracer = self.ray_tracers.get_phase_tracer(phase)
        traveltimes = await ray_tracer.get_travel_times(
            nodes=nodes,
            stations=stations,
            phase=phase,
        )

        if self.station_corrections:
            station_delays = await self.station_corrections.get_delays(
                [sta.nsl for sta in stations],
                nodes=nodes,
                phase=phase,
            )
            traveltimes += station_delays

        return traveltimes

    async def get_weights(
        self,
        stations: Sequence[Station],
        nodes: Sequence[Node],
        normalize_weights: bool = True,
        apply_waterlevel: float = 1e-3,
    ) -> np.ndarray:
        n_nodes = len(nodes)
        n_stations = len(stations)

        if self.distance_weights:
            weights = await self.distance_weights.get_weights(
                nodes=nodes,
                stations=stations,
            )
        else:
            weights = np.ones(shape=(n_nodes, n_stations), dtype=np.float32)

        if normalize_weights:
            with np.errstate(divide="ignore", invalid="ignore"):
                weights /= weights.sum(axis=1, keepdims=True)

        if apply_waterlevel:
            weights[weights < apply_waterlevel] = 0.0

        return weights

    async def search(
        self,
        images: WaveformImages,
        octree: Octree,
        split_nodes: list[Node] | None = None,
        stack: DelaySumReduce | None = None,
        n_threads: int = 0,
    ) -> tuple[list[EventDetection], Trace]:
        """Searches for events in the given traces.

        Args:
            images (WaveformImages): The waveform images to use for the search.
            octree (Octree | None, optional): The octree to use for the search.
                Defaults to None.
            split_nodes (list[Node] | None, optional): The nodes to split for this
                iteration. Defaults to None.
            stack (DelaySumReduce | None, optional): The stack to use for the search.
                If None, a new instance is created. Defaults to None.
            n_threads (int, optional): Number of threads for delay and sum.
                Defaults to 0

        Returns:
            tuple[list[EventDetection], Trace]: The event detections and the
                semblance traces used for the search.
        """
        if not stack:
            stack = DelaySumReduce(
                traces=images.get_traces(),
                start_time=images.start_time,
                end_time=images.end_time,
                padding=self.window_padding,
            )

        if split_nodes:
            new_nodes = []
            for node in split_nodes:
                new_nodes.extend(node.split())

            logger.info(
                "refining octree: split %d nodes to level %d (min size: %.1f m)",
                len(split_nodes),
                max(node.level for node in new_nodes + split_nodes),
                min(node.size for node in new_nodes + split_nodes),
            )
            stack.remove_nodes(split_nodes)
        else:
            new_nodes = octree.nodes

        weights = []
        traveltimes = []
        for image in images:
            image_traveltimes = await self.get_traveltimes(
                image.stations,
                nodes=new_nodes,
                phase=image.phase,
            )
            image_weights = await self.get_weights(
                stations=image.stations,
                nodes=new_nodes,
            )
            image_weights *= image.weight

            traveltimes.append(image_traveltimes)
            weights.append(image_weights)

        stack.add_nodes(
            nodes=new_nodes,
            traveltimes=np.hstack(traveltimes),
            weights=np.hstack(weights),
        )
        await stack.stack(n_threads=n_threads)

        if self.detection_threshold == "MAD":
            stack_max, _ = stack.get_stack(trim_padding=True)
            threshold = stats.median_abs_deviation(stack_max) * 10
            logger.debug("threshold MAD %g", self.detection_threshold)
        else:
            threshold = self.detection_threshold

        detection_idx, detection_semblance = await stack.find_peaks(
            height=float(threshold),
            prominence=float(threshold),
            distance=round(self.blinding.total_seconds() * images.sampling_rate),
        )

        if detection_idx.size == 0:
            return [], await stack.get_trace()

        # Split Octree nodes above a semblance threshold. Once octree for all detections
        # in frame
        node_candidates: set[Node] = set()
        _, stack_node_idx = stack.get_stack()

        for time_idx in detection_idx:
            node_idx = stack_node_idx[time_idx]
            source_node = octree.nodes[node_idx]

            if not source_node.is_leaf():
                logger.error("Hit a non-leaf node!")
                continue

            if self.ignore_boundary and source_node.is_inside_border(
                with_surface=self.ignore_boundary == "with_surface",
                border_width=self.ignore_boundary_width,
            ):
                logger.info("ignoring detection within boundary")
                continue
            node_candidates.add(source_node)
            # Getting neighbours is slow
            if self.neighbor_search:
                node_candidates.update(source_node.get_neighbours())

            # This iteration is also slow
            if self.semblance_density_search:
                snapshot = await stack.get_snapshot(time_idx)
                octree.map_semblance(snapshot)

                densest_node = max(
                    octree.leaf_nodes,
                    key=lambda n: n.semblance_density(),
                )
                if densest_node != source_node:
                    node_candidates.add(densest_node)
                    if self.neighbor_search:
                        node_candidates.update(densest_node.get_neighbours())

        node_candidates = {node for node in node_candidates if node.can_split()}

        # recursively refine nodes until all sources fall into the smallest nodes
        if node_candidates:
            logger.debug("detected %d event", detection_idx.size)
            return await self.search(
                images,
                octree,
                stack=stack,
                split_nodes=list(node_candidates),
                n_threads=n_threads,
            )

        detections = []
        logger.info("detected %d events", len(detection_idx))
        for time_idx, semblance_detection in zip(
            detection_idx, detection_semblance, strict=True
        ):
            time = stack.get_time_from_sample(time_idx)
            node_idx = stack_node_idx[time_idx]

            source_node = octree.nodes[node_idx]
            if self.ignore_boundary and source_node.is_inside_border(
                with_surface=self.ignore_boundary == "with_surface",
                border_width=self.ignore_boundary_width,
            ):
                continue

            semblance_event = await stack.get_snapshot(time_idx)
            octree.map_semblance(semblance_event)
            if self.node_interpolation:
                source_location = await octree.interpolate_max_semblance(source_node)
            else:
                source_location = source_node.as_location()

            detection = EventDetection(
                time=time,
                semblance=float(semblance_detection),
                distance_border=source_node.get_distance_border(),  # rename trough_distance
                n_stations=images.n_stations,
                **source_location.model_dump(),
            )

            # Attach modelled and picked arrivals to receivers
            for image in images:
                if not self.attach_arrivals:
                    break
                ray_tracer = self.ray_tracers.get_phase_tracer(image.phase)
                arrivals_model = ray_tracer.get_arrivals(
                    phase=image.phase,
                    event_time=time,
                    source=source_location,
                    receivers=image.stations,
                )

                if self.station_corrections:
                    for arrival, nsl in zip(
                        arrivals_model, image.stations.get_nsls(), strict=True
                    ):
                        delay = self.station_corrections.get_delay(
                            nsl, image.phase, node=source_node
                        )
                        if arrival and delay:
                            arrival.time += timedelta(seconds=delay)

                arrival_times = [arr.time if arr else None for arr in arrivals_model]

                arrivals_observed = image.search_phase_arrivals(
                    event_time=time,
                    modelled_arrivals=arrival_times,
                    threshold=self.pick_confidence_threshold,
                )

                phase_detections = [
                    PhaseDetection(
                        phase=image.phase,
                        model=modelled_time,
                        observed=obs,
                    )
                    if modelled_time
                    else None
                    for modelled_time, obs in zip(
                        arrivals_model,
                        arrivals_observed,
                        strict=True,
                    )
                ]
                detection.receivers.add(
                    stations=image.stations,
                    phase_arrivals=phase_detections,
                )
                detection.set_uncertainty(
                    DetectionUncertainty.from_event(
                        source_node=source_node,
                        octree=octree,
                    )
                )

            detections.append(detection)

        return detections, await stack.get_trace()
