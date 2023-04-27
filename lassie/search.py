from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from lassie.images import ImageFunction
from lassie.octree import Octree

if TYPE_CHECKING:
    from datetime import datetime

    from pyrocko.trace import Trace

    from lassie.config import Config
    from lassie.images.base import WaveformImages
    from lassie.models import Detection, Receivers

logger = logging.getLogger(__name__)


class Search:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.squirrel = config.get_squirrel()
        self.receivers = config.get_receivers()

        for tracer in config.ray_tracers:
            tracer.set_octree(config.octree)
            tracer.set_receivers(config.get_receivers())

    def search(self) -> None:
        config = self.config
        for batch in self.squirrel.chopper_waveforms(
            tmin=config.start_time.timestamp(),
            tmax=config.end_time.timestamp(),
            tinc=60,
            tpad=5,
            want_incomplete=True,
            codes=list((*nsl, "*") for nsl in self.receivers.all_nsl()),
        ):
            traces: list[Trace] = batch.traces
            if not traces:
                continue

            block = SearchBlock(
                traces,
                config.octree.copy(),
                self.receivers,
                config.image_functions,
            )
            block.calculate_images()


class SearchBlock:
    waveform_images: list[WaveformImages]
    detections: list[Detection]

    def __init__(
        self,
        traces: list[Trace],
        octree: Octree,
        receivers: Receivers,
        image_functions: list[ImageFunction],
    ) -> None:
        self.octree = octree
        self.traces = traces
        self.receivers = receivers
        self.image_functions = image_functions

        self.waveform_images = []
        self.clean_traces()

    def clean_traces(self) -> None:
        for tr in self.traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)

    def calculate_images(self) -> None:
        logger.info("calculating images")
        for image_func in self.image_functions:
            logger.debug("calculating images from %s", image_func.__class__.__name__)
            images = image_func.process_traces(self.traces)
            self.waveform_images.append(images)

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
        self.calculate_images()
