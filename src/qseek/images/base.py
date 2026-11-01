from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, ClassVar, Iterator, Literal, Sequence

import numpy as np
from pyrocko.io import save

from qseek.base import Model
from qseek.models.station import StationInventory, StationList
from qseek.stats import QueueStats
from qseek.utils import (
    NSL,
    QUEUE_SIZE,
    SDS_PYROCKO_SCHEME,
    PhaseDescription,
    datetime_now,
    prefetch,
)

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table

    from qseek.models.detection import EventDetection
    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)

PhaseName = Literal["P", "S"]


@dataclass
class ObservedArrival:
    phase: str
    time: datetime
    detection_value: float
    provider: str = ""


class Picker(Model):
    def pick_trace(
        self,
        trace: Trace,
        phase: PhaseDescription,
        event_time: datetime,
        modelled_arrival: datetime,
    ) -> ObservedArrival | None:
        """Pick a phase arrival in a single image function trace.

        Args:
            trace (Trace): Image function trace of a station.
            phase (PhaseDescription): Phase of the observed arrival.
            event_time (datetime): Time of the event, picks before it are rejected.
            modelled_arrival (datetime): Modelled arrival time to search around.

        Returns:
            ObservedArrival | None: Picked arrival, None if none found.
        """
        raise NotImplementedError

    def add_picks(
        self,
        detections: Sequence[EventDetection],
        images: WaveformImages,
    ) -> None:
        """Pick the observed arrivals of the detections' receivers.

        Picks are searched around the modelled arrival of each receiver's phase
        detection and attached as its observed arrival.

        Args:
            detections (Sequence[EventDetection]): Detections with modelled arrivals.
            images (WaveformImages): Images the detections were located from.
        """
        for image in images:
            # Stations can have multiple traces due to data gaps
            station_traces: dict[NSL, list[Trace]] = defaultdict(list)
            for tr in image.traces:
                station_traces[NSL(tr.network, tr.station, tr.location)].append(tr)

            for detection in detections:
                for receiver in detection.receivers:
                    arrival = receiver.phase_arrivals.get(image.phase)
                    if arrival is None:
                        continue
                    for trace in station_traces.get(receiver.nsl, ()):
                        pick = self.pick_trace(
                            trace,
                            image.phase,
                            detection.time,
                            arrival.model.time,
                        )
                        if pick is not None:
                            arrival.observed = pick
                            break


class ImageFunctionStats(QueueStats):
    time_per_batch: timedelta = timedelta()
    bytes_per_second: float = 0.0
    batch_size: int | None = None
    windows_per_second: float = 0.0

    _position = 40
    _show_header = False

    def _populate_table(self, table: Table) -> None:
        self._add_queue_row(table, "Phase annotation", self.bytes_per_second)
        if self.batch_size is not None:
            table.add_row(
                "Batch size",
                f"{self.batch_size} ({self.windows_per_second:.0f} windows/s)",
                style="dim",
            )


class ImageFunction(Model):
    image: Literal["base"] = "base"

    picker: Picker

    _stats: ClassVar[ImageFunctionStats] = ImageFunctionStats()

    @classmethod
    def get_subclasses(cls) -> tuple[type[ImageFunction], ...]:
        """Returns a tuple of all the subclasses of ImageFunction."""
        return tuple(cls.__subclasses__())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def prepare(self) -> None: ...

    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        """Process traces to generate image functions.

        Args:
            traces (list[Trace]): List of traces to process.

        Returns:
            list[WaveformImage]: List of image functions.
        """
        raise NotImplementedError

    def get_blinding(self) -> timedelta:
        """Blinding duration for the image function. Added to padded waveforms.

        Returns:
            timedelta: The blinding duration for the image function.
        """
        raise NotImplementedError("must be implemented by subclass")

    def get_phases(self) -> tuple[PhaseDescription, ...]:
        """Get the phases provided by the image function.

        Returns:
            tuple[PhaseDescription, ...]: The phases provided by the image function.
        """
        raise NotImplementedError("must be implemented by subclass")

    async def get_images(self, batch: WaveformBatch) -> WaveformImages:
        """Calculate the images of a waveform batch.

        Images without traces are skipped.

        Args:
            batch (WaveformBatch): Batch of waveforms.

        Returns:
            WaveformImages: Images of the batch.

        Raises:
            ValueError: If no image has traces.
        """
        images = WaveformImages(
            start_time=batch.start_time,
            end_time=batch.end_time,
        )
        logger.debug("calculating images from %s", self.name)
        for image in await self.process_traces(batch.traces):
            if not image.has_traces():
                logger.warning(
                    "no traces for %s image %s, skipping", self.name, image.phase
                )
                continue
            images.add_image(image)

        if not images.n_images:
            raise ValueError("no image has traces")
        return images

    def iter_images(
        self,
        batch_iterator: AsyncGenerator[WaveformBatch],
    ) -> AsyncGenerator[tuple[WaveformImages, WaveformBatch]]:
        """Iterate over images from batches.

        The images are calculated in a background task, ahead of the consumer.
        Batches whose images cannot be calculated due to a `ValueError` are
        skipped, other errors are raised to the consumer. The background task is
        cancelled when the consumer stops iterating.

        Args:
            batch_iterator (AsyncGenerator[WaveformBatch]): Async generator of
                batches.

        Returns:
            AsyncGenerator[tuple[WaveformImages, WaveformBatch]]: Images and their
                batch.
        """
        queue: asyncio.Queue[tuple[WaveformImages, WaveformBatch]] = asyncio.Queue(
            maxsize=QUEUE_SIZE
        )
        self._stats.set_queue(queue)
        logger.info("start pre-processing images, queue size %d", queue.maxsize)
        return prefetch(
            self._calculate_images(batch_iterator), queue, name="image function"
        )

    async def _calculate_images(
        self,
        batch_iterator: AsyncGenerator[WaveformBatch],
    ) -> AsyncGenerator[tuple[WaveformImages, WaveformBatch]]:
        stats = self._stats
        async with contextlib.aclosing(batch_iterator):
            async for batch in batch_iterator:
                if not batch.is_healthy():
                    logger.debug("unhealthy batch, skipping")
                    continue

                start_time = datetime_now()
                try:
                    images = await self.get_images(batch)
                except ValueError as e:
                    logger.warning("error processing images: %s", e)
                    continue
                stats.time_per_batch = datetime_now() - start_time
                stats.bytes_per_second = (
                    batch.nbytes / stats.time_per_batch.total_seconds()
                )
                yield images, batch
        logger.debug("image function finished")


@dataclass
class WaveformImage:
    image_function: str
    phase: PhaseDescription
    weight: float
    traces: list[Trace]
    detection_half_width: float

    _stations: StationList | None = None

    @property
    def stations(self) -> StationList:
        if self._stations is None:
            raise ValueError("Stations have not been set for this image.")
        return self._stations

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.delta_t

    @property
    def delta_t(self) -> float:
        return self.traces[0].deltat

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    def has_traces(self) -> bool:
        return bool(self.traces)

    def set_stations(self, stations: StationInventory) -> None:
        """Set stations from the image's available traces."""
        self._stations = StationList(stations.select_from_traces(self.traces))

    def get_trace_data(self) -> list[np.ndarray]:
        """Get all trace data in a list.

        Returns:
            list[np.ndarray]: List of numpy arrays.
        """
        if self._stations is None:
            raise ValueError("Stations must be set before getting trace data.")
        return [tr.ydata for tr in self.traces if tr.ydata is not None]

    def get_offsets(self, reference: datetime) -> np.ndarray:
        """Get traces timing offsets to a reference time in samples.

        Args:
            reference (datetime): Reference time.

        Returns:
            np.ndarray: Integer offset towards the reference for each trace.
        """
        if self._stations is None:
            raise ValueError("Stations must be set before getting trace data.")
        trace_tmins = np.fromiter((tr.tmin for tr in self.traces), float)
        return np.round((trace_tmins - reference.timestamp()) / self.delta_t).astype(
            np.int32
        )

    async def save_mseed(self, path: Path) -> None:
        """Save the image traces to disk.

        Args:
            path (Path): Path to save the traces.
        """
        save_traces = [tr.copy() for tr in self.traces]
        for tr in save_traces:
            tr.set_ydata((tr.ydata * 1e6).astype(np.int32))
        await asyncio.to_thread(
            save,
            save_traces,
            f"{path!s}/{SDS_PYROCKO_SCHEME}",
            append=True,
        )

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        snuffle(self.traces)


@dataclass
class WaveformImages:
    start_time: datetime
    end_time: datetime
    images: list[WaveformImage] = field(default_factory=list)
    _sampling_rate: float = 0.0

    @property
    def n_images(self) -> int:
        """Number of image functions."""
        return len(self.images)

    @property
    def n_stations(self) -> int:
        """Number of stations in the images."""
        return max(0, *(image.stations.n_stations for image in self if image.stations))

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the images."""
        return self._sampling_rate

    @property
    def duration(self) -> timedelta:
        """Duration of the images."""
        return self.end_time - self.start_time

    def add_image(self, image: WaveformImage) -> None:
        """Add an image to the collection.

        Args:
            image (WaveformImage): Image to add.
        """
        trace_sampling_rates = {1.0 / tr.deltat for tr in image.traces}
        if len(trace_sampling_rates) > 1:
            raise ValueError(
                f"Traces of image {image.phase} have different sampling rates "
                f"{', '.join(f'{sr:g}' for sr in sorted(trace_sampling_rates))} Hz. "
                "Resample the waveforms in the pre-processing."
            )
        self._sampling_rate = self._sampling_rate or image.sampling_rate
        if self._sampling_rate != image.sampling_rate:
            raise ValueError(
                f"Image sampling rate {image.sampling_rate} does not match existing "
                f"sampling rate {self._sampling_rate}"
            )
        self.images.append(image)

    def set_stations(self, stations: StationInventory) -> None:
        """Set the images stations.

        Args:
            stations (Stations): Stations to set.
        """
        for image in self:
            image.set_stations(stations)

    def cumulative_weight(self) -> float:
        """Get the cumulative weight of all images."""
        return sum(image.weight for image in self)

    def get_traces(self) -> list[Trace]:
        traces = []
        for img in self:
            traces += img.traces
        return traces

    def snuffle(self) -> None:
        """Open Pyrocko Snuffler on the image traces."""
        from pyrocko.trace import snuffle

        snuffle(self.get_traces())

    async def save_mseed(self, path: Path) -> None:
        """Save images to disk.

        Args:
            path (Path): Path to save the images.
        """
        logger.debug("saving images to %s", path)
        path.mkdir(exist_ok=True)
        for image in self:
            await image.save_mseed(path)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.images
