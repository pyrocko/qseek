from __future__ import annotations

import asyncio
import glob
import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Iterator

from pydantic import PositiveInt, PrivateAttr, conint, validator
from pyrocko.squirrel import Squirrel

from lassie.features import FeatureExtractors
from lassie.features.ground_motion import GroundMotionExtractor
from lassie.features.local_magnitude import LocalMagnitudeExtractor
from lassie.search.base import Search, SearchTraces
from lassie.utils import datetime_now, to_datetime

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch
    from pyrocko.trace import Trace

    from lassie.models.detection import EventDetection

logger = logging.getLogger(__name__)


class SquirrelPrefetcher:
    def __init__(self, iterator: Iterator[Batch], queue_size: int = 4) -> None:
        self.iterator = iterator
        self.queue: asyncio.Queue[Batch] = asyncio.Queue(maxsize=queue_size)

        self._task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self) -> None:
        logger.info("start prefetching squirrel data")
        while True:
            start = datetime_now()
            batch = await asyncio.to_thread(lambda: next(self.iterator, None))
            logger.debug("prefetched waveforms in %s", datetime_now() - start)
            if batch is None:
                logger.debug("squirrel prefetcher finished")
                await self.queue.wait(None)
                break
            await self.queue.put(batch)


class SquirrelSearch(Search):
    time_span: tuple[datetime | None, datetime | None] = (None, None)
    squirrel_environment: Path = Path(".")
    waveform_data: list[Path]
    waveform_prefetch_batches: PositiveInt = 4

    features: list[FeatureExtractors] = [
        GroundMotionExtractor(),
        LocalMagnitudeExtractor(),
    ]
    search_progress_time: datetime | None = None
    window_length_factor: conint(ge=5, le=100) = 10

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
            "searching time span from %s to %s (%s)",
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
            squirrel = Squirrel(str(self.squirrel_environment.expanduser()))
            paths = []
            for path in self.waveform_data:
                if "**" in str(path):
                    paths.extend(glob.glob(str(path.expanduser()), recursive=True))
                else:
                    paths.append(str(path.expanduser()))
            paths.extend((str(p.expanduser()) for p in self.stations.station_xmls))

            squirrel.add(paths, check=False)
            self._squirrel = squirrel
        return self._squirrel

    async def scan_squirrel(self) -> None:
        self.ray_tracers.prepare(self.octree, self.stations)
        self._init_ranges()
        squirrel = self.get_squirrel()

        window_increment = self.shift_range * self.window_length_factor
        logger.info("using trace window increment: %s", window_increment)

        start_time = self.start_time
        if self.search_progress_time:
            start_time = self.search_progress_time
            logger.info("continuing search from %s", start_time)

        iterator = squirrel.chopper_waveforms(
            tmin=start_time.timestamp(),
            tmax=self.end_time.timestamp(),
            tinc=window_increment.total_seconds(),
            tpad=self.window_padding.total_seconds(),
            want_incomplete=False,
            codes=[(*nsl, "*") for nsl in self.stations.get_all_nsl()],
        )
        prefetcher = SquirrelPrefetcher(iterator, self.waveform_prefetch_batches)

        batch_start_time = None
        batch_durations: Deque[timedelta] = deque(maxlen=25)
        while True:
            batch = await prefetcher.queue.get()
            if batch is None:
                logger.info("squirrel search finished")
                break

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
                if detection.in_bounds:
                    await self.add_features(detection)

                self._detections.add(detection)
                await self._new_detection.emit(detection)

            if detections:
                self._detections.dump_all()

            self.search_progress_time = window_end
            progress_file = self._rundir / "search_progress_time.txt"
            progress_file.write_text(str(self.search_progress_time))

            if batch_start_time is not None:
                batch_duration = datetime_now() - batch_start_time
                batch_durations.append(batch_duration)
                logger.info(
                    "window %d/%d took %s",
                    batch.i + 1,
                    batch.n,
                    batch_duration,
                )
                remaining_time = (
                    sum(batch_durations, timedelta())
                    / len(batch_durations)
                    * (batch.n - batch.i - 1)
                )
                logger.info(
                    "finish %s (remaining %s)",
                    datetime.now() + remaining_time,  # noqa: DTZ005
                    remaining_time,
                )
            batch_start_time = datetime_now()

            prefetcher.queue.task_done()

    async def add_features(self, event: EventDetection) -> None:
        squirrel = self.get_squirrel()

        for extractor in self.features:
            logger.info("adding features from %s", extractor.feature)
            await extractor.add_features(squirrel, event)
