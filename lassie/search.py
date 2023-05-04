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

        self.check_distances()

    def check_distances(self) -> None:
        distances = np.array(
            [
                node.as_location().distance_to(station)
                for node in self.octree
                for station in self.stations
            ]
        )
        print(f"Distances {distances.min()} - {distances.max()}")

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
            codes=list((*nsl, "*") for nsl in self.stations.get_all_nsl()),
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
        self.octree.reset()

        for image in images:
            logger.debug("stacking image %s", image.image_function.name)

            shifts = []
            weights = []

            ray_tracer = self.ray_tracers.get_phase_tracer(image.phase)
            stations = self.stations.select(image.get_all_nsl())

            for node in self.octree:
                traveltimes = ray_tracer.get_traveltimes(
                    phase=image.phase,
                    node=node,
                    stations=stations,
                )
                shifts.append(np.round(traveltimes / image.delta_t).astype(np.int32))
                weights.append(np.ones(image.n_traces))

            semblance, offsets = parstack.parstack(
                arrays=image.get_trace_data(),
                offsets=image.get_offsets(),
                shifts=np.array(shifts),
                weights=np.array(weights),
                dtype=np.float32,
                method=0,
            )

            semblance_argmax = parstack.argmax(
                semblance.astype(np.float64), nparallel=2
            )
            semblance_max = semblance.max(axis=0)

            time_idx = semblance_max.argmax()
            node_idx = semblance_argmax[time_idx]
            # plt.plot(semblance[:, time_idx])
            # plt.show()

            print(self.octree[node_idx].as_location())

            self.octree.add_semblance(semblance[:, time_idx])
            self.octree.plot()
            self.octree.plot_surface()
