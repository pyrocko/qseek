from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, PrivateAttr
from pyrocko import parstack

from lassie.images import ImageFunctions, WaveformImages
from lassie.models import Stations
from lassie.models.detection import EventDetection, EventDetections, PhaseDetection
from lassie.models.semblance import Semblance, SemblanceStats
from lassie.octree import NodeSplitError, Octree
from lassie.plot.octree import plot_octree_surface_tiles
from lassie.signals import Signal
from lassie.station_corrections import StationCorrections
from lassie.tracers import CakeTracer, ConstantVelocityTracer, RayTracers
from lassie.utils import PhaseDescription, Symbols, alog_call, time_to_path

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
    sampling_rate: SamplingRate = 50
    detection_threshold: PositiveFloat = 0.05
    detection_blinding: timedelta = timedelta(seconds=2.0)

    project_dir: Path = Path(".")

    octree: Octree = Octree()
    stations: Stations = Stations()
    ray_tracers: RayTracers = RayTracers(root=[ConstantVelocityTracer(), CakeTracer()])
    image_functions: ImageFunctions

    station_corrections: StationCorrections | None = None

    n_threads_parstack: int = Field(0, ge=0)
    n_threads_argmax: PositiveInt = 4

    node_split_threshold: float = Field(0.9, gt=0.0, lt=1.0)

    plot_octree_surface: bool = False

    # Overwritten at initialisation
    shift_range: timedelta = timedelta(seconds=0.0)
    window_padding: timedelta = timedelta(seconds=0.0)
    distance_range: tuple[float, float] = (0.0, 0.0)
    travel_time_ranges: dict[PhaseDescription, tuple[timedelta, timedelta]] = {}
    progress: SearchProgress = SearchProgress()

    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _detections: EventDetections = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    # Signals
    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())

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

        self.write_config()

        logger.info("created new rundir %s", rundir)

        self._detections = EventDetections(rundir=rundir)

    def write_config(self, path: Path | None = None) -> None:
        rundir = self._rundir
        path = path or rundir / "search.json"

        logger.debug("writing search config to %s", path)
        path.write_text(self.model_dump_json(indent=2, exclude_unset=True))

        logger.debug("dumping stations...")
        self.stations.dump_pyrocko_stations(rundir / "pyrocko-stations.yaml")
        self.stations.dump_csv(rundir / "stations.csv")

    @property
    def semblance_stats(self) -> SemblanceStats:
        return self.progress.semblance_stats

    def set_progress(self, time: datetime) -> None:
        self.progress.time_progress = time
        progress_file = self._rundir / "progress.json"
        progress_file.write_text(self.progress.model_dump_json())

    def init_search(self) -> None:
        """Initialise search."""
        file_logger = logging.FileHandler(self._rundir / "lassie.log")
        logging.root.addHandler(file_logger)

        # Grid/receiver distances
        distances = self.octree.distances_stations(self.stations)
        self.distance_range = (distances.min(), distances.max())

        # Timing ranges
        for phase, tracer in self.ray_tracers.iter_phase_tracer():
            traveltimes = tracer.get_traveltimes(phase, self.octree, self.stations)
            self.travel_time_ranges[phase] = (
                timedelta(seconds=np.nanmin(traveltimes)),
                timedelta(seconds=np.nanmax(traveltimes)),
            )
            logger.info(
                "shift ranges: %s / %s - %s", phase, *self.travel_time_ranges[phase]
            )

        # TODO: minimum shift is calculated on the coarse octree grid, which is
        # not necessarily the same as the fine grid used for semblance calculation
        shift_min = min(chain.from_iterable(self.travel_time_ranges.values()))
        shift_max = max(chain.from_iterable(self.travel_time_ranges.values()))
        self.shift_range = shift_max - shift_min

        self.window_padding = (
            self.shift_range
            + self.detection_blinding
            + self.image_functions.get_blinding()
        )

        logger.info(
            "source-station distances range: %.1f - %.1f m", *self.distance_range
        )
        logger.info("shift range %s", self.shift_range)
        logger.info("using trace window padding: %s", self.window_padding)
        self.write_config()

    def _plot_octree_surface(
        self,
        octree: Octree,
        time: datetime,
        detections: list[EventDetection] | None = None,
    ) -> None:
        logger.info("plotting octree surface...")
        filename = (
            self._rundir
            / "figures"
            / "octree_surface"
            / f"{time_to_path(time)}-nodes-{octree.n_nodes}.png"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        plot_octree_surface_tiles(octree, filename=filename, detections=detections)

    @classmethod
    def load_rundir(cls, rundir: Path) -> Self:
        search_file = rundir / "search.json"
        search = cls.model_validate_json(search_file.read_bytes())
        search._rundir = rundir
        search._detections = EventDetections.load_rundir(rundir)

        progress_file = rundir / "progress.json"
        if progress_file.exists():
            search.progress = SearchProgress.model_validate_json(
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
        self.traces = self.clean_traces(traces)

        self.start_time = start_time
        self.end_time = end_time

        self._images = {}

    @staticmethod
    def clean_traces(traces: list[Trace]) -> list[Trace]:
        """Remove empty or bad traces."""
        for tr in traces.copy():
            if not tr.ydata.size or not np.all(np.isfinite(tr.ydata)):
                logger.warning("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                traces.remove(tr)

        return traces

    def _n_samples_semblance(self) -> int:
        """Number of samples to use for semblance calculation, includes padding."""
        parent = self.parent
        window_padding = parent.window_padding
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

        traveltimes = ray_tracer.get_traveltimes(image.phase, octree, image.stations)

        if parent.station_corrections:
            station_delays = parent.station_corrections.get_delays(
                image.stations.get_all_nsl(), image.phase
            )
            traveltimes += station_delays[np.newaxis, :]

        traveltimes_bad = np.isnan(traveltimes)
        traveltimes[traveltimes_bad] = 0.0
        station_contribution = (~traveltimes_bad).sum(axis=1, dtype=float)

        shifts = np.round(-traveltimes / image.delta_t).astype(np.int32)
        weights = np.full_like(shifts, fill_value=image.weight, dtype=float)

        # Normalize by number of station contribution
        with np.errstate(divide="ignore", invalid="ignore"):
            weights /= station_contribution[:, np.newaxis]
        weights[traveltimes_bad] = 0.0

        semblance_data, offsets = await asyncio.to_thread(
            parstack.parstack,
            arrays=image.get_trace_data(),
            offsets=image.get_offsets(self.start_time - parent.window_padding),
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
            round(parent.window_padding.total_seconds() * sampling_rate)
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
        semblance.normalize(images.cumulative_weight())

        parent.semblance_stats.update(semblance.get_stats())
        logger.info("semblance stats: %s", parent.semblance_stats)

        detection_idx, detection_semblance = semblance.find_peaks(
            height=parent.detection_threshold,
            prominence=parent.detection_threshold,
            distance=round(parent.detection_blinding.total_seconds() * sampling_rate),
        )

        if parent.plot_octree_surface:
            octree.map_semblance(semblance.maximum_node_semblance())
            parent._plot_octree_surface(octree, time=self.start_time)

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
            if not octree.is_node_in_bounds(source_node):
                logger.info(
                    "source node is inside octree's absorbing boundary (%.1f m)",
                    source_node.distance_border,
                )
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
            logger.info(
                "%s new detection %s: %.5fE, %.5fN, %.1f m, semblance %.3f",
                Symbols.Target,
                detection.time,
                *detection.effective_lat_lon,
                detection.effective_depth,
                detection.semblance,
            )

            # detection.plot()

        # plot_octree_movie(octree, semblance, file=Path("/tmp/test.mp4"))
        if parent.plot_octree_surface:
            octree.map_semblance(semblance.maximum_node_semblance())
            parent._plot_octree_surface(
                octree, time=self.start_time, detections=detections
            )

        return detections, semblance.get_trace()
