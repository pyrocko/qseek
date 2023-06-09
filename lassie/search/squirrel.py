from __future__ import annotations

import asyncio
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from pydantic import PrivateAttr, validator
from pyrocko.squirrel import Squirrel
from pyrocko.squirrel.base import Batch

from lassie.features import FeatureExtractors
from lassie.features.ground_motion import GroundMotionExtractor
from lassie.features.local_magnitude import LocalMagnitudeExtractor
from lassie.search.base import Search, SearchTraces
from lassie.utils import alog_call, to_datetime

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from lassie.models.detection import EventDetection

logger = logging.getLogger(__name__)


class SquirrelSearch(Search):
    time_span: tuple[datetime | None, datetime | None] = (None, None)
    squirrel_environment: Path = Path(".")
    waveform_data: list[Path]

    features: list[FeatureExtractors] = [
        GroundMotionExtractor(),
        LocalMagnitudeExtractor(),
    ]

    _squirrel: Squirrel | None = PrivateAttr(None)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not all(self.time_span):
            sq_tmin, sq_tmax = self.get_squirrel().get_time_span(
                ["waveform", "waveform_promise"]
            )
            self.time_span = (
                self.time_span[0] or to_datetime(sq_tmin),
                self.time_span[1] or to_datetime(sq_tmax),
            )

        logger.info(
            "time span %s - %s (%s)",
            self.start_time,
            self.end_time,
            self.end_time - self.start_time,
        )

    @validator("time_span")
    def _validate_time_span(cls, range):  # noqa: N805
        if range[0] >= range[1]:
            raise ValueError(f"time range is invalid {range[0]} - {range[1]}")
        return range

    @property
    def start_time(self) -> datetime:
        return self.time_span[0]

    @property
    def end_time(self) -> datetime:
        return self.time_span[1]

    def get_squirrel(self) -> Squirrel:
        if not self._squirrel:
            squirrel = Squirrel(str(self.squirrel_environment))
            paths = []
            for path in self.waveform_data:
                if "**" in str(path):
                    paths.extend(glob.glob(str(path), recursive=True))
                else:
                    paths.append(str(path))
            paths.extend(map(str, self.stations.station_xmls))

            squirrel.add(paths, check=False)
            self._squirrel = squirrel
        return self._squirrel

    async def scan_squirrel(
        self,
        window_increment: timedelta | None = None,
    ) -> None:
        self._init_ranges()
        squirrel = self.get_squirrel()

        # TODO: too hardcoded
        window_increment = (
            window_increment or self.shift_range * 10 + 3 * self.window_padding
        )
        logger.info("using window increment: %s", window_increment)

        iterator = squirrel.chopper_waveforms(
            tmin=self.start_time.timestamp(),
            tmax=self.end_time.timestamp(),
            tinc=window_increment.total_seconds(),
            tpad=self.window_padding.total_seconds(),
            want_incomplete=False,
            codes=[(*nsl, "*") for nsl in self.stations.get_all_nsl()],
        )

        async def async_iterator() -> AsyncIterator[Batch]:
            @alog_call  # to log the call
            async def get_waveforms() -> Batch | None:
                return await asyncio.to_thread(lambda: next(iterator, None))

            while True:
                batch = await get_waveforms()
                if batch is None:
                    return
                yield batch

        async for batch in async_iterator():
            window_start = to_datetime(batch.tmin)
            window_end = to_datetime(batch.tmax)
            window_length = window_end - window_start
            logger.info(
                "searching time window %d/%d %s - %s",
                batch.i + 1,
                batch.n,
                window_start,
                window_end,
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
            detections, semblance_trace = await block.search()
            self._detections.add_semblance(semblance_trace)
            for detection in detections:
                await self.add_features(detection)
                self._detections.add(detection)
                await self._new_detection.emit(detection)

    async def add_features(self, event: EventDetection) -> None:
        squirrel = self.get_squirrel()

        for extractor in self.features:
            logger.info("adding features from %s", extractor.feature)
            await extractor.add_features(squirrel, event)
