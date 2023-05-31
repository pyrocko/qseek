from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    confloat,
    conint,
)
from pyrocko import parstack
from pyrocko.trace import Trace
from scipy import signal

from lassie.images import ImageFunctions
from lassie.images.base import WaveformImage
from lassie.models import Stations
from lassie.models.detection import Detections, EventDetection
from lassie.octree import NodeSplitError, Octree
from lassie.signals import Signal
from lassie.tracers import RayTracers
from lassie.utils import PhaseDescription, alog_call, to_datetime, to_path

if TYPE_CHECKING:
    from lassie.images import WaveformImages
    from lassie.octree import Node
    from lassie.tracers.base import RayTracer

logger = logging.getLogger(__name__)


MUTE_MEDIAN_LEVEL = 3.0


class Search(BaseModel):
    sampling_rate: confloat(ge=5.0, le=50.0) = 20.0
    detection_threshold: PositiveFloat = 0.1
    detection_blinding: timedelta = timedelta(seconds=2.0)

    project_dir: Path = Path(".")

    octree: Octree
    stations: Stations
    ray_tracers: RayTracers
    image_functions: ImageFunctions

    n_threads_parstack: conint(ge=0) = 0
    n_threads_argmax: PositiveInt = 2

    # Overwritten at initialisation
    shift_range: timedelta = timedelta(seconds=0.0)
    window_padding: timedelta = timedelta(seconds=0.0)
    distance_range: tuple[float, float] = (0.0, 0.0)
    travel_time_ranges: dict[PhaseDescription, tuple[timedelta, timedelta]] = {}
    semblance_stats: tuple[float, float, int] = (0.0, 0.0, 0)

    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _detections: Detections = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._init_ranges()

    def init_rundir(self, force=False) -> None:
        rundir = self.project_dir / self._config_stem or f"run-{to_path(self.created)}"
        self._rundir = rundir

        if rundir.exists() and not force:
            raise FileExistsError(f"Rundir {rundir} already exists")

        elif rundir.exists() and force:
            create_time = to_path(
                datetime.fromtimestamp(rundir.stat().st_ctime)  # noqa
            )
            rundir_backup = rundir.with_name(f"{rundir.name}.bak-{create_time}")
            rundir.rename(rundir_backup)
            logger.info("created backup of existing rundir to %s", rundir_backup)

        if not rundir.exists():
            rundir.mkdir()
        search_config = rundir / "search.json"
        search_config.write_text(self.json(indent=2))
        self.stations.dump_pyrocko_stations(rundir / "stations.yaml")

        logger.info("created new rundir %s", rundir)

        file_logger = logging.FileHandler(rundir / "lassie.log")
        logging.root.addHandler(file_logger)

        self._detections = Detections(rundir=rundir)

    @classmethod
    def load_rundir(cls, path: Path) -> Self:
        search = cls.parse_file(path / "search.json")
        search._rundir = path
        search._detections = Detections(rundir=path)
        return search

    def new_semblance_stats(self, mean: float, std: float) -> tuple[float, float]:
        prev_mean, prev_std, prev_count = self.semblance_stats
        count = prev_count + 1

        mean = (prev_mean * prev_count + mean) / count
        std = (prev_std * prev_count + std) / count
        self.semblance_stats = (mean, std, count)
        return self.semblance_stats[:2]

    def _init_ranges(self) -> None:
        # Grid/receiver distances
        distances = self.octree.distances_stations(self.stations)
        self.distance_range = (distances.min(), distances.max())

        # Timing ranges
        for phase, tracer in self.ray_tracers.iter_phase_tracer():
            traveltimes = tracer.get_traveltimes(phase, self.octree, self.stations)
            self.travel_time_ranges[phase] = (
                timedelta(seconds=traveltimes.min()),
                timedelta(seconds=traveltimes.max()),
            )
            logger.info("shifts: %s / %s - %s", phase, *self.travel_time_ranges[phase])

        shift_min = min(chain.from_iterable(self.travel_time_ranges.values()))
        shift_max = max(chain.from_iterable(self.travel_time_ranges.values()))
        self.shift_range = shift_max - shift_min

        self.window_padding = (
            self.shift_range
            + self.detection_blinding
            + self.image_functions.get_blinding()
        )

        self.ray_tracers.prepare(self.octree, self.stations)

        logger.info(
            "source-station distances range: %.1f - %.1f m", *self.distance_range
        )
        logger.info("using window padding: %s", self.window_padding)

    @property
    def padding_samples(self) -> int:
        return int(round(self.window_padding.total_seconds() * self.sampling_rate))

    @classmethod
    def parse_file(
        cls,
        path: str | Path,
    ) -> Self:
        model = super().parse_file(path)
        # Make relative paths absolute
        path = Path(path)
        base_dir = path.absolute().parent
        for name in model.__fields__:
            value = getattr(model, name)
            if isinstance(value, Path) and not value.absolute():
                setattr(model, name, value.relative_to(base_dir))
        model._config_stem = path.stem
        return model


class SearchTraces:
    _images: WaveformImages | None

    def __init__(
        self,
        parent: Search,
        traces: list[Trace],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        self.parent = parent
        self.traces = self.clean_traces(traces)

        self.start_time = start_time or to_datetime(min(tr.tmin for tr in self.traces))
        self.end_time = end_time or to_datetime(max(tr.tmax for tr in self.traces))

        self._images = None

    @staticmethod
    def clean_traces(traces: list[Trace]) -> list[Trace]:
        for tr in traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                traces.remove(tr)
        return traces

    @property
    def n_samples_semblance(self) -> int:
        window_padding = self.parent.window_padding
        time_span = (self.end_time + window_padding) - (
            self.start_time - window_padding
        )
        return int(round(time_span.total_seconds() * self.parent.sampling_rate))

    @alog_call
    async def calculate_semblance(
        self,
        octree: Octree,
        image: WaveformImage,
        ray_tracer: RayTracer,
        n_samples_semblance: int,
    ) -> np.ndarray:
        logger.debug("stacking image %s", image.image_function.name)
        parent = self.parent
        stations = parent.stations.select_from_traces(image.traces)

        traveltimes = ray_tracer.get_traveltimes(image.phase, octree, stations)
        traveltimes_bad = np.isnan(traveltimes)
        traveltimes[traveltimes_bad] = 0.0

        shifts = np.round(-traveltimes / image.delta_t).astype(np.int32)

        weights = np.ones_like(shifts)
        weights[traveltimes_bad] = 0.0

        semblance, offsets = await asyncio.to_thread(
            parstack.parstack,
            arrays=image.get_trace_data(),
            offsets=image.get_offsets(self.start_time - parent.window_padding),
            shifts=shifts,
            weights=weights,
            lengthout=n_samples_semblance,
            dtype=np.float32,
            method=0,
            nparallel=parent.n_threads_parstack,
        )

        # Normalize by number of station contribution
        station_contribution = (~traveltimes_bad).sum(axis=1)
        semblance /= station_contribution[:, np.newaxis]
        return semblance

    async def get_images(self) -> WaveformImages:
        if not self._images:
            images = await self.parent.image_functions.process_traces(self.traces)
            images.downsample(self.parent.sampling_rate)
            self._images = images
        return self._images

    async def search(
        self,
        octree: Octree | None = None,
    ) -> tuple[list[EventDetection], Trace]:
        parent = self.parent

        octree = octree or parent.octree.copy(deep=True)
        images = await self.get_images()

        semblance = np.zeros(
            (octree.n_nodes, self.n_samples_semblance),
            dtype=np.float32,
        )

        for image in images:
            semblance += await self.calculate_semblance(
                octree=octree,
                image=image,
                ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                n_samples_semblance=self.n_samples_semblance,
            )
        semblance /= images.n_images

        semblance_max = semblance.max(axis=0)
        semblance_node_idx = await asyncio.to_thread(
            parstack.argmax,
            semblance.astype(np.float64),
            nparallel=parent.n_threads_argmax,
        )

        # mask peaks by median muting and get mean/std
        semblance_median = np.median(semblance_max)
        peak_mask = semblance_max < (semblance_median * MUTE_MEDIAN_LEVEL)
        semblance_mean = float(np.mean(semblance_max[peak_mask]))
        semblance_std = float(np.std(semblance_max[peak_mask]))
        run_mean, run_std = parent.new_semblance_stats(semblance_mean, semblance_std)
        logger.info("semblance stats: mean %g, std %g", run_mean, run_std)

        detection_idx, _ = signal.find_peaks(
            semblance_max,
            height=run_mean + run_std * 9,
            prominence=run_std,
            distance=parent.detection_blinding.total_seconds() * parent.sampling_rate,
        )

        # Remove padding and shift peak detections
        if parent.padding_samples:
            padding_samples = parent.padding_samples

            semblance = semblance[:, padding_samples:-padding_samples]
            semblance_max = semblance_max[padding_samples:-padding_samples]
            semblance_node_idx = semblance_node_idx[padding_samples:-padding_samples]

            detection_idx -= padding_samples
            detection_idx = detection_idx[detection_idx >= 0]
            detection_idx = detection_idx[detection_idx < semblance_node_idx.size]

        semblance_trace = Trace(
            network="",
            station="semblance",
            tmin=self.start_time.timestamp(),
            deltat=1.0 / parent.sampling_rate,
            ydata=semblance_max,
        )

        if detection_idx.size == 0:
            return [], semblance_trace

        # Split Octree nodes above a semblance threshold. Once octree for all detections
        # in frame
        split_nodes: set[Node] = set()
        for idx in detection_idx:
            semblance_detection = semblance_max[idx]
            octree.map_semblance(semblance[:, idx])
            split_nodes.update(octree.get_nodes(semblance_detection * 0.9))

        try:
            new_nodes = [node.split() for node in split_nodes]
            sizes = set(node.size for node in chain.from_iterable(new_nodes))
            logger.info(
                "event detected - splitted %d octree nodes to %s m",
                len(split_nodes),
                ", ".join(f"{s:.1f}" for s in sizes),
            )
            return await self.search(octree)

        except NodeSplitError:
            logger.debug("reverting partial split")
            for node in split_nodes:
                node.reset()
            logger.debug("event detected - octree bottom %.1f m", octree.size_limit)

        detections = []
        for idx in detection_idx:
            time = self.start_time + timedelta(seconds=idx / parent.sampling_rate)
            semblance_detection = semblance_max[idx]

            octree.map_semblance(semblance[:, idx])

            idx = semblance_node_idx[idx]
            source_node = octree[idx].as_location()

            detection = EventDetection.construct(
                time=time,
                semblance=float(semblance_detection),
                octree=octree.copy(deep=True),
                **source_node.dict(),
            )
            detections.append(detection)
            logger.info(
                "new detection %s: %.5fE, %.5fN, %.1f m, semblance %.3f",
                detection.time,
                *detection.effective_lat_lon,
                detection.effective_depth,
                detection.semblance,
            )

        return detections, semblance_trace
