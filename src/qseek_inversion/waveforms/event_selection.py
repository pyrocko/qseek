from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Literal, TypeVar

import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr
from pyrocko.io import save
from pyrocko.trace import Trace
from scipy import signal

from qseek.images.images import ImageFunctions, WaveformImages
from qseek.pre_processing.module import PreProcessing
from qseek.search import Search
from qseek.utils import NSLC
from qseek.waveforms.base import WaveformBatch
from qseek_inversion.waveforms.base import WaveformSelection

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _chunk_iterator(events: list[_T], size: int) -> Generator[list[_T], Any, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(events), size):
        yield events[i : i + size]


class EventTraceGroup:
    event: EventDetection
    n_samples: int
    sampling_rate: float
    traces: dict[NSLC, Trace]

    def __init__(self, event: EventDetection, traces: list[Trace]):
        n_samples = {tr.ydata.size for tr in traces}
        if len(n_samples) != 1:
            raise ValueError("Traces have different number of samples")
        deltats = {tr.deltat for tr in traces}
        if len(deltats) != 1:
            raise ValueError("Traces have different sampling rates")

        self.event = event
        self.sampling_rate = 1.0 / deltats.pop()
        self.n_samples = n_samples.pop()
        self.traces = {NSLC(*tr.nslc_id): tr for tr in traces}
        if len(traces) != len(self.traces):
            raise ValueError("Traces have duplicate NSLC ids")

    def get_nslcs(self) -> set[NSLC]:
        return set(self.traces.keys())


class ConcatenatedTrace:
    nslc: NSLC
    sampling_rate: float
    data: np.ndarray

    def __init__(self, nslc: NSLC, sampling_rate: float):
        self.nslc = nslc
        self.sampling_rate = sampling_rate
        self.data = np.zeros(0, dtype=np.float32)

    def add_data(self, new_data: np.ndarray, taper: float = 0.1) -> None:
        taper_window = signal.windows.tukey(new_data.size, alpha=taper)
        n_samples_overlap = int(new_data.size * taper / 2)

        n_samples_add = new_data.size - n_samples_overlap
        if self.data.size == 0:
            n_samples_add = new_data.size

        self.data = np.pad(
            self.data,
            (0, n_samples_add),
            mode="constant",
            constant_values=0.0,
        )
        self.data[-new_data.size :] += signal.detrend(new_data) * taper_window

    def get_trace(self) -> Trace:
        return Trace(
            tmin=datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp(),
            ydata=self.data.astype(np.int32),
            deltat=1.0 / self.sampling_rate,
            network=self.nslc.network,
            station=self.nslc.station,
            location=self.nslc.location,
            channel=self.nslc.channel,
        )


class EventWaveformsSelection(WaveformSelection):
    waveforms: Literal["EventSelection"] = "EventSelection"

    import_rundir: Path = Field(
        ...,
        description="Path to the qseek run directory to import events from.",
    )

    number_events: PositiveInt = Field(
        default=200,
        description="Number of events to use for inversion."
        "The `N` events are ordered by semblance and selected.",
    )
    events_per_batch: PositiveInt = Field(
        default=25,
        description="Number of events to include in each waveform batch.",
    )
    seconds_before_event: PositiveFloat = Field(
        default=2.0,
        description="Number of seconds to include before the first arrival of the event.",
    )
    seconds_after_event: PositiveFloat = Field(
        default=2.0,
        description="Number of seconds to include after the last arrival of the event.",
    )
    taper: PositiveFloat = Field(
        default=0.1,
        le=0.5,
        ge=0.0,
        description="Fraction of the waveform to taper with a Tukey window.",
    )
    sampling_rate: PositiveFloat = Field(
        default=100.0,
        description="Sampling rate in Hz to resample the waveforms to.",
    )

    _events: list[EventDetection] = PrivateAttr(default_factory=list)

    _pre_processing: PreProcessing = PrivateAttr()
    _image_functions: ImageFunctions = PrivateAttr()
    _squirrel: Squirrel = PrivateAttr()

    async def prepare(
        self,
        rundir: Path | None = None,
    ) -> None:
        # TODO: This is the simplest selection process. Improve this.
        # E.g., avoid selecting events that are too close to each other.
        # Or select events based on depth.
        # Or select events based on magnitude.
        # Or select events based on location uncertainty.

        search = Search.load_rundir(self.import_rundir)
        await search.pre_processing.prepare()
        await search.image_functions.prepare(search.stations, search.octree)

        self._pre_processing = search.pre_processing
        self._image_functions = search.image_functions
        self._squirrel = search.data_provider.get_squirrel()

        search.stations.sanitize_stations()
        search.stations.filter_stations(search.data_provider.available_nsls())
        self.stations = search.stations

        events_sorted = sorted(
            search.catalog.events,
            key=lambda ev: ev.semblance,
            reverse=True,
        )
        logger.info(
            "selected %d events out of %d available events",
            min(self.number_events, len(events_sorted)),
            search.catalog.n_events,
        )
        events_sorted = [ev for ev in events_sorted if ev.in_bounds]
        self._events = events_sorted[: self.number_events]

        if rundir is not None:
            await self.export(rundir, self._squirrel)

    async def export(self, outpath: Path, squirrel: Squirrel) -> None:
        logger.info("exporting selected events and traces to %s", outpath)
        self.export_event_csv(outpath / "selected_events.csv")
        await self.export_mseed(
            squirrel,
            outpath / "concatenated_traces",
        )

    async def export_mseed(self, squirrel: Squirrel, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        traces = await self._get_contatenated_traces(self._events, squirrel)
        save(
            traces,
            str(outdir / "%(station)s.%(station)s.%(location)s-concatenated.mseed"),
        )
        logger.info("exported concatenated event traces to %s", outdir)

    def export_event_csv(self, outpath: Path) -> None:
        """Export selected events to CSV.

        Args:
            outpath (Path): Path to output CSV file.
        """
        for ev in self._events:
            ev.export_csv_line(outpath)
        logger.info("exported selected events to %s", outpath)

    async def get_images(self, window_padding: timedelta) -> list[WaveformImages]:
        batches = await self.get_batches(
            self._squirrel,
            edge_padding=window_padding,  # Why 2x?
        )
        batch_images = []

        for batch in batches:
            batch = await self._pre_processing.process_batch(batch)
            images = await self._image_functions.get_images(batch)
            images.set_stations(self.stations)
            batch_images.append(images)
        return batch_images

    async def get_batches(
        self,
        squirrel: Squirrel,
        edge_padding: timedelta = timedelta(seconds=0),
    ) -> list[WaveformBatch]:
        batches = []

        for events in _chunk_iterator(self._events, self.events_per_batch):
            traces = await self._get_contatenated_traces(events, squirrel)
            for trace in traces:
                padding_seconds = edge_padding.total_seconds()
                trace.extend(
                    tmin=trace.tmin - padding_seconds,
                    tmax=trace.tmax + padding_seconds,
                    fillmethod="zeros",
                )

            start_times = {tr.tmin for tr in traces}
            end_times = {tr.tmax for tr in traces}
            if len(end_times) != 1 or len(start_times) != 1:
                raise ValueError("Traces have different start or end times")

            batch = WaveformBatch(
                traces=traces,
                start_time=datetime.fromtimestamp(start_times.pop(), tz=timezone.utc),
                end_time=datetime.fromtimestamp(end_times.pop(), tz=timezone.utc),
                i_batch=0,
            )
            batches.append(batch)
        return batches

    async def _get_contatenated_traces(
        self,
        events: list[EventDetection],
        squirrel: Squirrel,
    ) -> list[Trace]:
        concatenated_traces: dict[NSLC, ConcatenatedTrace] = {}
        event_groups: list[EventTraceGroup] = []

        for event in events:
            traces = await self._get_event_traces(
                event,
                squirrel,
                seconds_before=self.seconds_before_event,
                seconds_after=self.seconds_after_event,
            )
            for nslc in traces.get_nslcs():
                if nslc not in concatenated_traces:
                    concatenated_traces[nslc] = ConcatenatedTrace(
                        nslc=nslc,
                        sampling_rate=self.sampling_rate,
                    )
            event_groups.append(traces)

        for event_grp in event_groups:
            for concat_tr in concatenated_traces.values():
                event_tr = event_grp.traces.get(concat_tr.nslc, None)
                if event_tr is None:
                    data = np.zeros(event_grp.n_samples, dtype=np.float32)
                else:
                    data: np.ndarray = event_tr.ydata
                concat_tr.add_data(data, taper=self.taper)
        return [concat_tr.get_trace() for concat_tr in concatenated_traces.values()]

    async def _get_event_traces(
        self,
        event: EventDetection,
        squirrel: Squirrel,
        seconds_before: float = 2.0,
        seconds_after: float = 2.0,
    ) -> EventTraceGroup:
        traces = await event.receivers.get_waveforms(
            squirrel=squirrel,
            seconds_before=seconds_before,
            seconds_after=seconds_after,
            want_incomplete=False,
            crop_receivers=False,
        )
        if not traces:
            raise ValueError(f"No traces found for event {event.time}")

        for tr in traces:
            if tr.deltat != 1.0 / self.sampling_rate:
                tr.resample(1.0 / self.sampling_rate)

        start_time = max(tr.tmin for tr in traces)
        end_time = min(tr.tmax for tr in traces)

        for tr in traces:
            tr.chop(start_time, end_time, snap=(math.ceil, math.floor))

        min_nsamples = min(tr.ydata.size for tr in traces)
        for tr in traces:
            if tr.ydata.size > min_nsamples:
                tr.set_ydata(tr.ydata[:min_nsamples].copy())

        return EventTraceGroup(event=event, traces=traces)
