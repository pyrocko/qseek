from __future__ import annotations

import logging
import pprint
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from pyrocko import parstack
from pyrocko.trace import Trace
from scipy import signal

from lassie.images.base import WaveformImage
from lassie.models.detection import Detection, Detections
from lassie.octree import Octree
from lassie.tracers.base import RayTracer
from lassie.utils import to_datetime

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.squirrel import Squirrel

    from lassie.images import ImageFunctions
    from lassie.models import Stations
    from lassie.tracers import RayTracers

logger = logging.getLogger(__name__)


class Search:
    distance_range: tuple[float, float]
    travel_time_ranges: dict[str, tuple[float, float]]

    def __init__(
        self,
        octree: Octree,
        stations: Stations,
        ray_tracers: RayTracers,
        image_functions: ImageFunctions,
        sampling_rate: float = 20,
        detection_threshold: float = 0.1,
        detection_blinding: timedelta = timedelta(seconds=2.0),
    ) -> None:
        self.octree = octree
        self.stations = stations
        self.ray_tracers = ray_tracers
        self.image_functions = image_functions

        self.sampling_rate = sampling_rate
        self.detection_blinding = detection_blinding
        self.detection_threshold = detection_threshold

        self.travel_time_ranges = {}
        for phase, tracer in self.ray_tracers.iter_phase_tracer():
            traveltimes = tracer.get_traveltimes(phase, self.octree, self.stations)
            self.travel_time_ranges[phase] = (traveltimes.min(), traveltimes.max())

        self.detections = Detections()

        self.check_ranges()

    def check_ranges(self) -> None:
        self.distances = self.octree.distances_stations(self.stations)
        self.distance_range = (self.distances.min(), self.distances.max())

        shift_ranges = np.array([times for times in self.travel_time_ranges.values()])
        shift_min = timedelta(seconds=shift_ranges.min())
        shift_max = timedelta(seconds=shift_ranges.max())
        self.shift_range = shift_max - shift_min

        self.window_padding = (
            self.shift_range
            + self.detection_blinding
            + self.image_functions.get_blinding()
        )
        self.window_padding = timedelta(seconds=1.0)

        logger.info("source-station distances range: %s", self.distance_range)
        logger.info("source shift ranges:\n%s", pprint.pformat(self.travel_time_ranges))
        logger.info("window padding: %s s", self.window_padding)

    @property
    def padding_samples(self) -> int:
        return int(self.window_padding.total_seconds() * self.sampling_rate)

    def scan_squirrel(
        self,
        squirrel: Squirrel,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        window_increment: timedelta | None = None,
    ) -> None:
        window_increment = (
            window_increment or self.shift_range * 10 + 3 * self.window_padding
        )
        logger.info("window increment: %s", window_increment)

        for batch in squirrel.chopper_waveforms(
            tmin=start_time.timestamp() if start_time else None,
            tmax=end_time.timestamp() if end_time else None,
            tinc=window_increment.total_seconds(),
            tpad=self.window_padding.total_seconds(),
            want_incomplete=False,
            codes=list((*nsl, "*") for nsl in self.stations.get_all_nsl()),
        ):
            window_start = to_datetime(batch.tmin)
            window_end = to_datetime(batch.tmax)
            window_length = window_end - window_start
            logger.info(
                "searching time window %d/%d %s - %s (%s)",
                batch.i + 1,
                batch.n,
                window_start,
                window_end,
                window_length,
            )

            traces: list[Trace] = batch.traces
            if not traces:
                logger.warning("window is empty")
                continue

            if window_start > window_end or window_length < 2 * self.window_padding:
                logger.warning("window length is too short")
                continue

            block = SearchTraces(
                parent=self,
                traces=traces,
                start_time=window_start,
                end_time=window_end,
            )
            self.detections.append(block.search())


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
        return int(time_span.total_seconds() * self.parent.sampling_rate)

    def calculate_semblance(
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

        semblance, offsets = parstack.parstack(
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

    def search(self) -> Detections:
        parent = self.parent

        images = parent.image_functions.process_traces(self.traces)
        images.downsample(parent.sampling_rate)

        semblance = np.zeros(
            (self.octree.n_nodes, self.n_samples_semblance),
            dtype=np.float32,
        )

        for image in images:
            semblance += self.calculate_semblance(
                image=image,
                ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                n_samples_semblance=self.n_samples_semblance,
            )
        semblance /= images.n_images

        semblance_max = semblance.max(axis=0)
        semblance_node_idx = parstack.argmax(semblance.astype(np.float64), nparallel=2)

        peak_idx, _ = signal.find_peaks(
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

            peak_idx -= padding_samples
            peak_idx = peak_idx[peak_idx >= 0]
            peak_idx = peak_idx[peak_idx < semblance_node_idx.size]

        detections = Detections()
        for idx in peak_idx:
            node_idx = semblance_node_idx[idx]
            self.octree.add_semblance(semblance[:, idx])

            source_node = self.octree[node_idx].as_location()

            detection = Detection.construct(
                time=self.start_time,
                detection_peak=semblance_max.max(),
                octree=self.octree.copy(),
                **source_node.dict(),
            )
            detections.add_detection(detection)

            detection.plot_surface()

        self.semblance_trace = Trace(
            network="",
            station="semblance",
            tmin=self.start_time.timestamp(),
            deltat=1.0 / parent.sampling_rate,
            ydata=semblance_max,
        )

        return detections
