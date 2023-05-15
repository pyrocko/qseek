from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, PositiveFloat, PrivateAttr, confloat
from pyrocko import parstack
from pyrocko.trace import Trace
from scipy import signal

from lassie.images import ImageFunctions
from lassie.images.base import WaveformImage
from lassie.models import Stations
from lassie.models.detection import Detection, Detections
from lassie.octree import Octree
from lassie.tracers import RayTracers
from lassie.tracers.base import RayTracer
from lassie.utils import PhaseDescription, to_datetime, to_path

logger = logging.getLogger(__name__)


class Search(BaseModel):
    sampling_rate: confloat(ge=5.0, le=50.0) = 20.0
    detection_threshold: PositiveFloat = 0.1
    detection_blinding: timedelta = timedelta(seconds=2.0)

    project_dir: Path = Path(".")

    octree: Octree
    stations: Stations
    ray_tracers: RayTracers
    image_functions: ImageFunctions

    # Overwritten at initialisation
    shift_range: timedelta = timedelta(seconds=0.0)
    window_padding: timedelta = timedelta(seconds=0.0)
    distance_range: tuple[float, float] = (0.0, 0.0)
    travel_time_ranges: dict[PhaseDescription, tuple[timedelta, timedelta]] = {}

    _created: datetime = PrivateAttr(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    _detections: Detections = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._init_ranges()

    def init_rundir(self, force=False) -> None:
        rundir = self.project_dir / self._config_stem or f"run-{to_path(self._created)}"
        self._rundir = rundir

        if rundir.exists() and not force:
            raise EnvironmentError(f"Rundir {rundir} already exists")

        elif rundir.exists() and force:
            backup_time = to_path(datetime.now(tz=timezone.utc))
            rundir_backup = rundir.with_name(f"{rundir.name}-bak-{backup_time}")
            rundir.rename(rundir_backup)
            logger.info("created backup of existing rundir to %s", rundir_backup)

        if not rundir.exists():
            rundir.mkdir()
        search_config = rundir / "search.json"
        search_config.write_text(self.json(indent=2))

        self._detections = Detections(rundir=rundir)
        logger.info("created new rundir %s", rundir)

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
            logger.info("shifts: %s / %s - %s", phase, *self.travel_time_ranges[phase])

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
        logger.info("window padding: %s", self.window_padding)

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
    def __init__(
        self,
        parent: Search,
        traces: list[Trace],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        self.parent = parent
        self.octree = parent.octree.copy()
        self.traces = self.clean_traces(traces)

        self.start_time = start_time or to_datetime(min(tr.tmin for tr in self.traces))
        self.end_time = end_time or to_datetime(max(tr.tmax for tr in self.traces))

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

    async def calculate_semblance(
        self,
        image: WaveformImage,
        ray_tracer: RayTracer,
        n_samples_semblance: int,
    ) -> np.ndarray:
        logger.debug("stacking image %s", image.image_function.name)
        parent = self.parent
        stations = parent.stations.select_from_traces(image.traces)

        traveltimes = ray_tracer.get_traveltimes(image.phase, parent.octree, stations)
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
            nparallel=6,
        )

        # Normalize by number of station contribution
        station_contribution = (~traveltimes_bad).sum(axis=1)
        semblance /= station_contribution[:, np.newaxis]
        return semblance

    async def search(self) -> tuple[list[Detection], Trace]:
        parent = self.parent

        images = await parent.image_functions.process_traces(self.traces)
        images.downsample(parent.sampling_rate)

        semblance = np.zeros(
            (self.octree.n_nodes, self.n_samples_semblance),
            dtype=np.float32,
        )

        for image in images:
            semblance += await self.calculate_semblance(
                image=image,
                ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                n_samples_semblance=self.n_samples_semblance,
            )
        semblance /= images.n_images

        semblance_max = semblance.max(axis=0)
        semblance_node_idx = await asyncio.to_thread(
            parstack.argmax,
            semblance.astype(np.float64),
            nparallel=2,
        )

        detection_idx, _ = signal.find_peaks(
            semblance_max,
            height=parent.detection_threshold,
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

        detections = []
        for idx in detection_idx:
            node_idx = semblance_node_idx[idx]
            self.octree.add_semblance(semblance[:, idx])

            time = self.start_time + timedelta(seconds=idx * parent.sampling_rate)
            semblance_detection = semblance_max[idx]

            source_node = self.octree[node_idx].as_location()

            detection = Detection.construct(
                time=time,
                semblance=float(semblance_detection),
                octree=self.octree.copy(),
                **source_node.dict(),
            )
            detections.append(detection)

        semblance_trace = Trace(
            network="",
            station="semblance",
            tmin=self.start_time.timestamp(),
            deltat=1.0 / parent.sampling_rate,
            ydata=semblance_max,
        )

        return detections, semblance_trace
