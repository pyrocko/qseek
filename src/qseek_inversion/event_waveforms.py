from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pyrocko.trace import Trace
from scipy import signal

from qseek.utils import NSLC
from qseek.waveforms.base import WaveformBatch

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.catalog import EventCatalog
    from qseek.models.detection import EventDetection

logger = logging.getLogger(__name__)


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

    def add_data(self, new_data: np.ndarray, taper_percent: float = 0.1) -> None:
        taper = signal.windows.tukey(new_data.size, alpha=taper_percent)
        n_samples_overlap = int(new_data.size * taper_percent / 2)

        n_samples_add = new_data.size - n_samples_overlap
        if self.data.size == 0:
            n_samples_add = new_data.size

        self.data = np.pad(
            self.data,
            (0, n_samples_add),
            mode="constant",
            constant_values=0.0,
        )
        self.data[-new_data.size :] += signal.detrend(new_data) * taper

    def get_trace(self) -> Trace:
        return Trace(
            tmin=0.0,
            ydata=self.data.astype(np.int32),
            deltat=1.0 / self.sampling_rate,
            network=self.nslc.network,
            station=self.nslc.station,
            location=self.nslc.location,
            channel=self.nslc.channel,
        )


class EventWaveforms(BaseModel):
    number_events: PositiveInt = Field(
        default=200,
        description="Number of events to use for inversion."
        "The `N` events are ordered by semblance and selected.",
    )
    sampling_rate: PositiveFloat = Field(
        default=100.0,
        description="Sampling rate in Hz to resample the waveforms to.",
    )

    def select_events(self, catalog: EventCatalog) -> list[EventDetection]:
        # TODO: This is the simplest selection process. Improve this.
        # E.g., avoid selecting events that are too close to each other.
        # Or select events based on depth.
        # Or select events based on magnitude.
        # Or select events based on location uncertainty.
        events_sorted = sorted(
            catalog.events,
            key=lambda ev: ev.semblance,
            reverse=True,
        )
        logger.info(
            "Selected %d events out of %d available events",
            min(self.number_events, len(events_sorted)),
            catalog.n_events,
        )
        return events_sorted[: self.number_events]

    async def get_contatenated_traces(
        self,
        catalog: EventCatalog,
        squirrel: Squirrel,
    ) -> list[Trace]:
        concatenated_traces: dict[NSLC, ConcatenatedTrace] = {}
        event_groups: list[EventTraceGroup] = []

        for event in self.select_events(catalog):
            traces = await self._get_event_traces(event, squirrel)
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
                concat_tr.add_data(data)
        return [concat_tr.get_trace() for concat_tr in concatenated_traces.values()]

    async def get_batch(
        self, catalog: EventCatalog, squirrel: Squirrel
    ) -> WaveformBatch:
        traces = await self.get_contatenated_traces(catalog, squirrel)
        tr_end_times = {tr.tmax for tr in traces}
        if len(tr_end_times) != 1:
            raise ValueError("Traces have different end times")

        return WaveformBatch(
            traces=traces,
            start_time=datetime.fromtimestamp(0.0, tz=timezone.utc),
            end_time=datetime.fromtimestamp(tr_end_times.pop(), tz=timezone.utc),
            i_batch=0,
        )

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
