from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pyrocko import parstack

from lassie.octree import Octree
from lassie.utils import to_datetime

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.images import ImageFunctions, WaveformImages
    from lassie.models import Detection, Stations
    from lassie.tracers import RayTracers

logger = logging.getLogger(__name__)


class Search:
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

    def scan_squirrel(
        self,
        squirrel: Squirrel,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        time_span = squirrel.get_time_span()
        start_time = start_time or to_datetime(time_span[0])
        end_time = end_time or to_datetime(time_span[1])

        for batch in squirrel.chopper_waveforms(
            tmin=start_time.timestamp(),
            tmax=end_time.timestamp(),
            tinc=60.0,
            tpad=0,
            want_incomplete=False,
            codes=list((*nsl, "*") for nsl in self.stations.all_nsl()),
        ):
            traces: list[Trace] = batch.traces
            if not traces:
                continue

            block = SearchTraces(
                traces,
                self.octree.copy(),
                self.stations,
                self.ray_tracers,
                self.image_functions,
            )
            block.search()


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
    ) -> None:
        self.octree = octree
        self.traces = self.clean_traces(traces)

        self.stations = stations
        self.ray_tracers = ray_tracers
        self.image_functions = image_functions

        self.trace_data = np.array([tr.ydata for tr in traces])
        self.offsets = np.zeros(self.n_traces)

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    @staticmethod
    def clean_traces(traces: list[Trace]) -> list[Trace]:
        for tr in traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                traces.remove(tr)
        return traces

    def search(
        self,
    ) -> Octree:
        images = self.image_functions.process_traces(self.traces)

        for image in images:
            shifts = []
            weights = []
            ray_tracer = self.ray_tracers.get_phase_tracer(image.phase)

            for node in self.octree:
                traveltimes = ray_tracer.get_traveltimes(
                    phase=image.phase,
                    node=node,
                    stations=self.stations,
                )
                shifts.append(
                    np.round(traveltimes / image.sampling_rate).astype(np.int32)
                )
                weights.append(np.ones(self.n_traces))

            print(
                "trace_data",
                self.trace_data.shape,
                "offsets",
                self.offsets.shape,
                "shifts",
                np.array(shifts).T.shape,
                "weights",
                np.array(weights).shape,
            )

            result, offsets = parstack.parstack(
                arrays=self.trace_data,
                offsets=self.offsets,
                shifts=np.array(shifts).T,
                weights=np.array(weights),
                dtype=np.float32,
                method=0,
            )
            print(result)
