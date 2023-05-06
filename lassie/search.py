from __future__ import annotations

import logging
import pprint
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from pyrocko import parstack
from scipy import signal

from lassie.images.base import WaveformImage
from lassie.models.detection import Detection, Detections
from lassie.octree import Octree
from lassie.utils import to_datetime

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.images import ImageFunctions, WaveformImages
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
    ) -> None:
        self.octree = octree
        self.stations = stations
        self.ray_tracers = ray_tracers
        self.image_functions = image_functions
        self.travel_time_ranges = {}
        self.detections = Detections()

        self.check_ranges()

    def check_ranges(self) -> None:
        self.distances = self.octree.distances_stations(self.stations)
        self.distance_range = (self.distances.min(), self.distances.max())

        for phase in self.ray_tracers.get_available_phases():
            tracer = self.ray_tracers.get_phase_tracer(phase)

            traveltimes = tracer.get_traveltimes(phase, self.octree, self.stations)
            self.travel_time_ranges[phase] = (traveltimes.min(), traveltimes.max())

        shift_ranges = np.array([times for times in self.travel_time_ranges.values()])
        shift_min = shift_ranges.min()
        shift_max = shift_ranges.max()
        shift_span = shift_max - shift_min

        self.peak_search_prominence = shift_span
        self.window_padding = (
            shift_span
            + self.peak_search_prominence / 2
            + self.image_functions.get_max_blinding_seconds()
        )
        self.window_padding = 1.0

        self.window_increment = shift_span * 10 + 3 * self.window_padding

        logger.info("source-station distances range: %s", self.distance_range)
        logger.info("source shift ranges:\n%s", pprint.pformat(self.travel_time_ranges))
        logger.info("window length: %.2f s", self.window_increment)
        logger.info("window padding: %.2f s", self.window_padding)

    def scan_squirrel(
        self,
        squirrel: Squirrel,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        for batch in squirrel.chopper_waveforms(
            tmin=start_time.timestamp() if start_time else None,
            tmax=end_time.timestamp() if end_time else None,
            tinc=self.window_increment,
            tpad=self.window_padding,
            want_incomplete=False,
            codes=list((*nsl, "*") for nsl in self.stations.get_all_nsl()),
        ):
            traces: list[Trace] = batch.traces
            if not traces:
                continue

            logger.info(
                "searching time block %s - %s",
                to_datetime(batch.tmin),
                to_datetime(batch.tmax),
            )

            block = SearchTraces(
                traces,
                self.octree,
                self.stations,
                self.ray_tracers,
                self.image_functions,
                start_time=to_datetime(batch.tmin),
                end_time=to_datetime(batch.tmax),
                padding_seconds=self.window_padding,
            )
            self.detections.append(block.search())


class SearchTraces:
    waveform_images: WaveformImages | None
    detections: list[Detection]

    def __init__(
        self,
        traces: list[Trace],
        octree: Octree,
        stations: Stations,
        ray_tracers: RayTracers,
        image_functions: ImageFunctions,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        padding_seconds: float = 0.0,
    ) -> None:
        self.octree = octree
        self.traces = self.clean_traces(traces)
        self.stations = stations
        self.ray_tracers = ray_tracers
        self.image_functions = image_functions

        self.start_time = start_time or to_datetime(min(tr.tmin for tr in self.traces))
        self.end_time = end_time or to_datetime(max(tr.tmax for tr in self.traces))
        self.padding_seconds = timedelta(seconds=padding_seconds)

        self.n_traces = len(self.traces)

    @staticmethod
    def clean_traces(traces: list[Trace]) -> list[Trace]:
        for tr in traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                traces.remove(tr)
        return traces

    def calculate_semblance(self, image: WaveformImage, lengthout: int) -> np.ndarray:
        logger.debug("stacking image %s", image.image_function.name)

        stations = self.stations.select_from_traces(image.traces)

        ray_tracer = self.ray_tracers.get_phase_tracer(image.phase)
        traveltimes = ray_tracer.get_traveltimes(image.phase, self.octree, stations)
        traveltimes_bad = np.isnan(traveltimes)
        traveltimes[traveltimes_bad] = 0.0

        shifts = np.round(-traveltimes / image.delta_t).astype(np.int32)

        weights = np.ones_like(shifts)
        weights[traveltimes_bad] = 0.0

        semblance, offsets = parstack.parstack(
            arrays=image.get_trace_data(),
            offsets=image.get_offsets(self.start_time),
            shifts=shifts,
            weights=weights,
            lengthout=lengthout,
            dtype=np.float32,
            method=0,
            nparallel=6,
        )
        return semblance

    def search(self) -> Detections:
        self.octree.reset()
        detections = Detections()

        images = self.image_functions.process_traces(self.traces)
        for image in images:
            image.chop(
                self.start_time + self.padding_seconds,
                self.end_time - self.padding_seconds,
            )

        max_samples = images.get_max_samples()
        semblance = np.zeros(
            (self.octree.n_nodes, max_samples),
            dtype=np.float32,
        )

        for image in images:
            semblance += self.calculate_semblance(image, max_samples)

        semblance_max = semblance.max(axis=0)
        semblance_node_idx = parstack.argmax(semblance.astype(np.float64), nparallel=2)

        peak_idx, _ = signal.find_peaks(semblance_max, height=10.0, distance=200)

        plt.plot(semblance_max)
        plt.scatter(peak_idx, semblance_max[peak_idx])
        plt.show()
        for idx in peak_idx:
            node_idx = semblance_node_idx[idx]

            node = self.octree[node_idx].as_location()
            detection = Detection.construct(
                time=self.start_time,
                detection_peak=semblance_max.max(),
                **node.dict(),
            )
            detections.add_detection(detection)
            self.octree.add_semblance(semblance[:, idx])
            self.octree.plot_surface()

        return detections
