from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
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

from lassie.models.detection import Detections, EventDetection, PhaseDetection
from lassie.models.semblance import Semblance, SemblanceStats
from lassie.octree import NodeSplitError, Octree
from lassie.signals import Signal
from lassie.utils import (
    ANSI,
    PhaseDescription,
    Symbols,
    alog_call,
    time_to_path,
    to_datetime,
)

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from lassie.images import ImageFunctions, WaveformImages
    from lassie.images.base import WaveformImage
    from lassie.models import Stations
    from lassie.octree import Node
    from lassie.tracers import RayTracers
    from lassie.tracers.base import RayTracer

logger = logging.getLogger(__name__)


MUTE_MEDIAN_LEVEL = 3.0


class Search(BaseModel):
    sampling_rate: confloat(ge=5.0, le=50.0) | None = None
    detection_threshold: PositiveFloat = 0.05
    detection_blinding: timedelta = timedelta(seconds=2.0)

    project_dir: Path = Path(".")

    octree: Octree
    stations: Stations
    ray_tracers: RayTracers
    image_functions: ImageFunctions

    n_threads_parstack: conint(ge=0) = 0
    n_threads_argmax: PositiveInt = 4

    # Overwritten at initialisation
    shift_range: timedelta = timedelta(seconds=0.0)
    window_padding: timedelta = timedelta(seconds=0.0)
    distance_range: tuple[float, float] = (0.0, 0.0)
    travel_time_ranges: dict[PhaseDescription, tuple[timedelta, timedelta]] = {}
    semblance_stats: SemblanceStats = SemblanceStats()

    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _detections: Detections = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.ray_tracers.prepare(self.octree, self.stations)

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
        search_config = rundir / "search.json"
        search_config.write_text(self.json(indent=2))
        self.stations.dump_pyrocko_stations(rundir / "pyrocko-stations.yaml")

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
            logger.info(
                "shift ranges: %s / %s - %s", phase, *self.travel_time_ranges[phase]
            )

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
        logger.info("using trace window padding: %s", self.window_padding)

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
    _images: dict[float | None, WaveformImages]

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

        self._images = {}

    @staticmethod
    def clean_traces(traces: list[Trace]) -> list[Trace]:
        for tr in traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warning("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                traces.remove(tr)
        return traces

    def get_n_samples_semblance(self, sampling_rate) -> int:
        window_padding = self.parent.window_padding
        time_span = (self.end_time + window_padding) - (
            self.start_time - window_padding
        )
        return int(round(time_span.total_seconds() * sampling_rate))

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

        traveltimes = ray_tracer.get_traveltimes(image.phase, octree, image.stations)
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
                raise TypeError("sampling rate has to be a float")
            images_resampled = deepcopy(self._images[None])

            logger.debug("downsampling images to %g Hz", sampling_rate)
            images_resampled.downsample(sampling_rate, max_normalize=True)
            self._images[sampling_rate] = images_resampled

        return self._images[sampling_rate]

    async def search(
        self,
        octree: Octree | None = None,
    ) -> tuple[list[EventDetection], Trace]:
        parent = self.parent

        octree = octree or parent.octree.copy(deep=True)

        sampling_rate = parent.sampling_rate
        if not sampling_rate:
            max_velocity = parent.ray_tracers.get_velocity_max()
            smallest_node = octree.get_smallest_node_size()
            sampling_rate = round(max_velocity / (smallest_node / 2), ndigits=-1)
            sampling_rate = min(sampling_rate, 100.0)
            logger.info(
                "fastest velocity %g m/s, smallest node %g m, sampling rate %g Hz",
                max_velocity,
                smallest_node,
                sampling_rate,
            )

        images = await self.get_images(sampling_rate=sampling_rate)
        logger.info("stacking traces at sampling rate %.1f Hz", sampling_rate)

        padding_samples = int(
            round(parent.window_padding.total_seconds() * sampling_rate)
        )

        semblance = Semblance(
            n_nodes=octree.n_nodes,
            n_samples=self.get_n_samples_semblance(sampling_rate),
            start_time=self.start_time,
            sampling_rate=sampling_rate,
            padding_samples=padding_samples,
        )

        for image in images:
            semblance.add(
                await self.calculate_semblance(
                    octree=octree,
                    image=image,
                    ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                    n_samples_semblance=self.get_n_samples_semblance(sampling_rate),
                )
            )
        semblance.normalize(images.n_images)

        parent.semblance_stats.update(semblance.get_stats())
        logger.info("semblance stats: %s", parent.semblance_stats)

        detection_idx, detection_semblance = semblance.find_peaks(
            height=parent.detection_threshold,
            prominence=parent.detection_threshold,
            distance=round(parent.detection_blinding.total_seconds() * sampling_rate),
        )

        if detection_idx.size == 0:
            return [], semblance.get_trace()

        # Split Octree nodes above a semblance threshold. Once octree for all detections
        # in frame
        split_nodes: set[Node] = set()
        for idx, semblance_detection in zip(detection_idx, detection_semblance):
            octree.map_semblance(semblance.semblance[:, idx])
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
        for idx, semblance_detection in zip(detection_idx, detection_semblance):
            time = self.start_time + timedelta(seconds=idx / sampling_rate)

            octree.map_semblance(semblance.semblance[:, idx])

            idx = (await semblance.maximum_node_idx())[idx]
            node = octree[idx]
            if octree.is_node_absorbed(node):
                logger.info(
                    "source node is inside octree's absorbing boundary (%.1f m)",
                    node.distance_border,
                )
                continue

            source_node = node.as_location()

            detection = EventDetection(
                time=time,
                semblance=float(semblance_detection),
                distance_border=node.distance_border,
                **source_node.dict(),
            )

            # Attach receivers with modelled and picked arrivals
            for image in await self.get_images(sampling_rate=None):
                ray_tracer = parent.ray_tracers.get_phase_tracer(image.phase)
                arrivals_model = ray_tracer.get_arrivals(
                    phase=image.phase,
                    event_time=time,
                    source=source_node,
                    receivers=image.stations.stations,
                )
                arrivals_observed = image.search_phase_arrivals(
                    modelled_arrivals=[arr.time for arr in arrivals_model]
                )
                phase_detections = [
                    PhaseDetection(phase=image.phase, model=model, observed=observed)
                    for model, observed in zip(
                        arrivals_model, arrivals_observed, strict=True
                    )
                ]
                detection.receivers.add_receivers(
                    stations=image.stations.stations,
                    phase_arrivals=phase_detections,
                )

            detections.append(detection)
            logger.info(
                "%s%s new detection %s: %.5fE, %.5fN, %.1f m, semblance %.3f%s",
                ANSI.Bold,
                Symbols.Target,
                detection.time,
                *detection.effective_lat_lon,
                detection.effective_depth,
                detection.semblance,
                ANSI.Reset,
            )

            # detection.plot()

        # plot_octree_movie(octree, semblance, file=Path("/tmp/test.mp4"))

        return detections, semblance.get_trace()
