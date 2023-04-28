from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from lassie.octree import Octree
from lassie.utils import to_datetime

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.images import ImageFunctions, WaveformImages
    from lassie.models import Detection, Receivers
    from lassie.tracers import RayTracers

logger = logging.getLogger(__name__)


class Search:
    def __init__(
        self,
        octree: Octree,
        receivers: Receivers,
        ray_tracers: RayTracers,
        image_functions: ImageFunctions,
    ) -> None:
        self.octree = octree
        self.receivers = receivers
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
            tinc=60,
            tpad=5,
            want_incomplete=True,
            codes=list((*nsl, "*") for nsl in self.receivers.all_nsl()),
        ):
            traces: list[Trace] = batch.traces
            if not traces:
                continue

            block = SearchTraces(
                traces,
                self.octree.copy(),
                self.receivers,
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
        receivers: Receivers,
        ray_tracers: RayTracers,
        image_functions: ImageFunctions,
    ) -> None:
        self.octree = octree
        self.traces = traces
        self.receivers = receivers
        self.ray_tracers = ray_tracers
        self.image_functions = image_functions

        self.waveform_images = None
        self.clean_traces()

    def clean_traces(self) -> None:
        for tr in self.traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)

    def stack_window(
        self,
        start_time: datetime,
        length: float,
        padding: float,
    ) -> Octree:
        if not self.waveform_images:
            raise ValueError("Images have not been calculated.")

        return self.octree

    def search(self) -> None:
        self.images = self.image_functions.process_traces(self.traces)
