from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from lru import LRU
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from scipy import signal
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection
    from qseek.utils import NSL
    from qseek.waveforms.base import WaveformProvider

logger = logging.getLogger(__name__)

PhaseType = Literal["P", "S"]
# Order of the zero-phase Butterworth bandpass, applied forward and backward
FILTER_ORDER = 4
# Filter padding before and after the windows in periods of the lowest frequency
PADDING_PERIODS = 3.0
# Upper corner of the bandpass at most this fraction of the Nyquist frequency
MAX_NYQUIST_FRACTION = 0.9
# A P window clipped before the S wave must keep this fraction of its length
MIN_WINDOW_FRACTION = 0.5
MIN_SAMPLES = 8


class PhaseWindow(BaseModel):
    """Correlation window of a phase around the pick or modeled arrival."""

    model_config = ConfigDict(extra="forbid")

    seconds_before: NonNegativeFloat = Field(
        description="Start of the window before the arrival in s.",
    )
    seconds_after: PositiveFloat = Field(
        description="End of the window after the arrival in s.",
    )
    max_lag: PositiveFloat = Field(
        description="Maximum lag between the two events in s. Lags at this limit "
        "are rejected, the correlation maximum lies outside.",
    )
    components: str = Field(
        min_length=1,
        description="Orientation codes of the correlated channels, e.g. `Z` or "
        "`NE12`. The normalized correlations of all components are stacked.",
    )

    @field_validator("components")
    @classmethod
    def _unique_components(cls, components: str) -> str:
        if len(set(components)) != len(components):
            raise ValueError(f"components {components!r} repeat an orientation code")
        return components


class DifferentialTime(NamedTuple):
    nsl: NSL
    phase: PhaseType
    time: float  # travel time of event 1 minus event 2 in s
    coefficient: float


class CorrelationEvent(NamedTuple):
    id: int
    detection: EventDetection
    origin: float  # timestamp of the origin time the travel times refer to
    # timestamps of the window anchors: picks or modeled arrivals
    arrivals: dict[tuple[NSL, PhaseType], float]


class WaveformCache:
    """LRU cache of the filtered waveforms of events, limited in bytes."""

    def __init__(self, max_bytes: int) -> None:
        self._cache: LRU = LRU(1_000_000)
        self._sizes: dict[object, int] = {}
        self.max_bytes = max_bytes
        self.n_bytes = 0

    def get(self, key: object) -> list[Trace] | None:
        return self._cache.get(key)

    def __setitem__(self, key: object, traces: list[Trace]) -> None:
        size = sum(tr.ydata.nbytes for tr in traces)
        self.n_bytes -= self._sizes.get(key, 0)
        self._cache[key] = traces
        self._sizes[key] = size
        self.n_bytes += size
        while self.n_bytes > self.max_bytes and len(self._cache) > 1:
            old_key, _ = self._cache.popitem()  # least recently used
            self.n_bytes -= self._sizes.pop(old_key)

    def hit_rate(self) -> float:
        hits, misses = self._cache.get_stats()
        return hits / max(1, hits + misses)


class CrossCorrelation(BaseModel):
    """Differential times from the cross-correlation of the waveforms of close events.

    The waveforms of two events are correlated in windows around the P and S
    arrivals at their common stations: the pick, or the modeled arrival at stations
    without a pick. The window of the first event is the template, the window of
    the second event extends by `max_lag` on both sides. All waveforms are
    bandpass filtered with the same zero-phase Butterworth filter.
    """

    model_config = ConfigDict(extra="forbid")

    bandpass: tuple[PositiveFloat, PositiveFloat] = Field(
        default=(1.0, 15.0),
        description="Corner frequencies of the bandpass filter in Hz, applied to "
        "all channels. The upper corner is limited to 90% of the Nyquist frequency.",
    )
    window_p: PhaseWindow = Field(
        default=PhaseWindow(
            seconds_before=0.1, seconds_after=0.5, max_lag=0.2, components="Z"
        ),
        description="Window of the P phase. It ends before the window of the S "
        "phase starts, so close stations correlate the P wave only.",
    )
    window_s: PhaseWindow = Field(
        default=PhaseWindow(
            seconds_before=0.2, seconds_after=1.0, max_lag=0.3, components="NE12"
        ),
        description="Window of the S phase.",
    )
    min_correlation: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Minimum correlation coefficient of a differential time. "
        "Its weight in `dt.cc` is the squared coefficient.",
    )
    max_separation: PositiveFloat = Field(
        default=2000.0,
        description="Maximum separation of a correlated event pair in m.",
    )
    max_neighbors: PositiveInt = Field(
        default=20,
        description="Maximum number of nearest neighbors correlated per event.",
    )
    min_observations: PositiveInt = Field(
        default=4,
        description="Minimum number of differential times of an event pair in `dt.cc`.",
    )
    modeled_arrivals: bool = Field(
        default=True,
        description="Correlate at stations without a pick, around the modeled "
        "arrival. Their differential times do not depend on the modeled arrival, "
        "only the window does.",
    )
    channels: list[str] | None = Field(
        default=None,
        description="Priority of the band and instrument codes, e.g. "
        '`["HH", "EH"]`. `null` uses the channels of the waveform provider.',
    )
    cache_size: ByteSize = Field(
        default=ByteSize(2 * 1024**3),
        description="Size of the cache of the filtered waveforms. Events that do "
        "not fit are loaded again.",
    )
    n_parallel: PositiveInt = Field(
        default=8,
        description="Number of events processed in parallel.",
    )

    @model_validator(mode="after")
    def _check_bandpass(self) -> CrossCorrelation:
        low, high = self.bandpass
        if low >= high:
            raise ValueError(f"bandpass {low}-{high} Hz: low corner must be below high")
        return self

    def get_window(self, phase: PhaseType) -> PhaseWindow:
        return self.window_p if phase == "P" else self.window_s

    @property
    def padding(self) -> float:
        """Padding for the filter before and after the windows in s."""
        return PADDING_PERIODS / self.bandpass[0]

    def select_pairs(self, events: list[CorrelationEvent]) -> list[tuple[int, int]]:
        """Event pairs to correlate: the nearest neighbors within `max_separation`.

        Returns:
            list[tuple[int, int]]: Pairs of indices into `events`, the first index
                is the smaller one.
        """
        if len(events) < 2:
            return []
        reference = events[0].detection
        coordinates = np.array(
            [ev.detection.offset_from(reference) for ev in events], dtype=float
        )
        tree = KDTree(coordinates)
        k = min(self.max_neighbors + 1, len(events))
        distances, neighbors = tree.query(
            coordinates, k=k, distance_upper_bound=self.max_separation
        )
        pairs: set[tuple[int, int]] = set()
        for idx, (row_distances, row_neighbors) in enumerate(
            zip(np.atleast_2d(distances), np.atleast_2d(neighbors), strict=True)
        ):
            for distance, neighbor in zip(row_distances, row_neighbors, strict=True):
                if not np.isfinite(distance) or neighbor == idx:
                    continue
                pairs.add((min(idx, int(neighbor)), max(idx, int(neighbor))))
        return sorted(pairs)

    async def correlate(
        self,
        events: list[CorrelationEvent],
        waveform_provider: WaveformProvider,
    ) -> dict[tuple[int, int], list[DifferentialTime]]:
        """Correlate the waveforms of close event pairs.

        Args:
            events: The events with the times of their arrivals.
            waveform_provider: The provider of the waveforms, prepared.

        Returns:
            dict[tuple[int, int], list[DifferentialTime]]: Differential times of the
                event pairs, keyed by the event IDs, with at least
                `min_observations` times.
        """
        pairs = self.select_pairs(events)
        neighbors: dict[int, list[int]] = defaultdict(list)
        for first, second in pairs:
            neighbors[first].append(second)
        logger.info(
            "correlating %d event pairs closer than %.0f m, bandpass %g-%g Hz",
            len(pairs),
            self.max_separation,
            *self.bandpass,
        )

        cache = WaveformCache(int(self.cache_size))
        loading: dict[int, asyncio.Task[list[Trace]]] = {}
        semaphore = asyncio.Semaphore(self.n_parallel)
        results: dict[tuple[int, int], list[DifferentialTime]] = {}
        stats: Counter[str] = Counter()
        n_done = 0
        n_report = max(1, len(neighbors) // 10)

        async def load(idx: int) -> list[Trace]:
            event = events[idx]
            try:
                traces = await self.load_waveforms(event, waveform_provider, stats)
                cache[event.detection.uid] = traces
            finally:
                loading.pop(idx, None)
            return traces

        async def get_waveforms(idx: int) -> list[Trace]:
            traces = cache.get(events[idx].detection.uid)
            if traces is not None:
                return traces
            if idx not in loading:
                loading[idx] = asyncio.create_task(load(idx))
            # other events wait for the same load, do not cancel it for them
            return await asyncio.shield(loading[idx])

        async def correlate_neighbors(idx: int) -> None:
            nonlocal n_done
            async with semaphore:
                others = neighbors[idx]
                waveforms = await asyncio.gather(
                    *(get_waveforms(i) for i in (idx, *others))
                )
                pair_times = await asyncio.to_thread(
                    lambda: [
                        self.correlate_pair(
                            events[idx], events[other], waveforms[0], traces
                        )
                        for other, traces in zip(others, waveforms[1:], strict=True)
                    ]
                )
            for other, times in zip(others, pair_times, strict=True):
                if len(times) >= self.min_observations:
                    results[(events[idx].id, events[other].id)] = times
            n_done += 1
            if n_done % n_report == 0 or n_done == len(neighbors):
                logger.info(
                    "correlated %d/%d events, %d pairs with differential times,"
                    " cache %.0f MB, hit rate %.0f%%",
                    n_done,
                    len(neighbors),
                    len(results),
                    cache.n_bytes / 1024**2,
                    cache.hit_rate() * 100,
                )

        await asyncio.gather(*(correlate_neighbors(idx) for idx in sorted(neighbors)))
        self.log_stats(stats, len(results))
        return dict(sorted(results.items()))

    def log_stats(self, stats: Counter[str], n_pairs: int) -> None:
        """Log the traces that were dropped, warn if no data are left."""
        dropped = {
            "no_data": "without data covering the windows and the filter padding",
            "nyquist": f"with a Nyquist frequency below {self.bandpass[0]:g} Hz",
            "filter": "that cannot be filtered",
        }
        for key, reason in dropped.items():
            if stats[key]:
                logger.info("dropped %d traces %s", stats[key], reason)
        if not stats["filtered"]:
            logger.warning(
                "no waveforms to correlate: check the waveform archive, `channels`,"
                " the `components` of the windows and `bandpass`"
            )
        elif not n_pairs:
            logger.warning(
                "no event pair has %d differential times with a correlation of at"
                " least %g",
                self.min_observations,
                self.min_correlation,
            )

    async def load_waveforms(
        self,
        event: CorrelationEvent,
        waveform_provider: WaveformProvider,
        stats: Counter[str] | None = None,
    ) -> list[Trace]:
        """Load and filter the waveforms of an event around its arrivals."""
        spans: dict[NSL, tuple[float, float]] = {}
        for (nsl, phase), time in event.arrivals.items():
            window = self.get_window(phase)
            start = time - window.seconds_before - window.max_lag
            end = time + window.seconds_after + window.max_lag
            if nsl in spans:
                start = min(start, spans[nsl][0])
                end = max(end, spans[nsl][1])
            spans[nsl] = (start, end)
        if not spans:
            return []

        event_receivers = event.detection.receivers
        receivers = [event_receivers.get_receiver(nsl) for nsl in spans]
        max_window = max(
            w.seconds_before + w.seconds_after + w.max_lag
            for w in (self.window_p, self.window_s)
        )
        try:
            traces = await event_receivers.get_waveforms(
                waveform_provider,
                seconds_before=max_window + self.padding,
                seconds_after=max_window + self.padding,
                receivers=receivers,
                channels=self.channels,
                crop_traces=False,
            )
        except OSError as exc:
            logger.warning("cannot load the waveforms of event %d: %s", event.id, exc)
            return []
        return await asyncio.to_thread(self.filter_waveforms, traces, spans, stats)

    def filter_waveforms(
        self,
        traces: list[Trace],
        spans: dict[NSL, tuple[float, float]],
        stats: Counter[str] | None = None,
    ) -> list[Trace]:
        """Cut the traces to their spans plus padding, filter and remove the padding.

        Traces with gaps in their span are dropped. `stats` counts the traces that
        were filtered and dropped.
        """
        from qseek.utils import NSL

        stats = Counter() if stats is None else stats
        components = self.window_p.components + self.window_s.components
        low, high = self.bandpass
        padding = self.padding
        filtered = []
        for tr in traces:
            span = spans.get(NSL(*tr.nslc_id[:3]))
            if span is None or tr.channel[-1:] not in components:
                continue
            # a few samples more, the windows start and end between samples
            start = span[0] - 2 * tr.deltat
            end = span[1] + 2 * tr.deltat
            # a segment covering the span with its padding has no gap in it
            if tr.tmin > start - padding or tr.tmax < end + padding:
                stats["no_data"] += 1
                continue
            high_corner = min(high, MAX_NYQUIST_FRACTION * 0.5 / tr.deltat)
            if high_corner <= low:
                stats["nyquist"] += 1
                continue
            tr = tr.chop(start - padding, end + padding, inplace=False)
            try:
                data = signal.detrend(tr.ydata.astype(np.float64))
                taper = min(1.0, 2 * padding / (tr.tmax - tr.tmin))
                data *= signal.windows.tukey(data.size, alpha=taper)
                data = signal.sosfiltfilt(
                    butterworth(round(1.0 / tr.deltat, 6), low, high_corner), data
                )
            except ValueError as exc:
                logger.debug("cannot filter %s: %s", ".".join(tr.nslc_id), exc)
                stats["filter"] += 1
                continue
            tr.set_ydata(data.astype(np.float32))
            tr.chop(start, end)
            filtered.append(tr)
            stats["filtered"] += 1
        return filtered

    def correlate_pair(
        self,
        event_1: CorrelationEvent,
        event_2: CorrelationEvent,
        traces_1: list[Trace],
        traces_2: list[Trace],
    ) -> list[DifferentialTime]:
        """Differential times of an event pair at their common stations.

        The travel time difference is the difference of the matched window starts,
        each relative to its origin time: the windows only select the waveform.
        """
        channels_1 = channel_map(traces_1)
        channels_2 = channel_map(traces_2)
        times: list[DifferentialTime] = []
        for (nsl, phase), arrival_1 in event_1.arrivals.items():
            arrival_2 = event_2.arrivals.get((nsl, phase))
            if arrival_2 is None:
                continue
            window = self.get_window(phase)
            seconds_after = window.seconds_after
            if phase == "P":
                # end the P window before the S window of both events starts
                for event, arrival in ((event_1, arrival_1), (event_2, arrival_2)):
                    arrival_s = event.arrivals.get((nsl, "S"))
                    if arrival_s is not None:
                        seconds_after = min(
                            seconds_after,
                            arrival_s - self.window_s.seconds_before - arrival,
                        )
                length = window.seconds_before + window.seconds_after
                if window.seconds_before + seconds_after < MIN_WINDOW_FRACTION * length:
                    continue

            pairs = [
                (channels_1[(nsl, comp)], channels_2[(nsl, comp)])
                for comp in window.components
                if (nsl, comp) in channels_1 and (nsl, comp) in channels_2
            ]
            pairs = [
                (tr_1, tr_2)
                for tr_1, tr_2 in pairs
                if tr_1.channel == tr_2.channel
                and math.isclose(tr_1.deltat, tr_2.deltat, rel_tol=1e-6)
            ]
            if not pairs:
                continue
            result = correlate_window(
                pairs,
                start_1=arrival_1 - window.seconds_before,
                start_2=arrival_2 - window.seconds_before - window.max_lag,
                duration=window.seconds_before + seconds_after,
                max_lag=window.max_lag,
            )
            if result is None:
                continue
            matched_1, matched_2, coefficient = result
            if coefficient < self.min_correlation:
                continue
            time = (matched_1 - event_1.origin) - (matched_2 - event_2.origin)
            times.append(DifferentialTime(nsl, phase, time, coefficient))
        return times


@lru_cache(maxsize=64)
def butterworth(sampling_rate: float, low: float, high: float) -> np.ndarray:
    return signal.butter(
        FILTER_ORDER, (low, high), btype="bandpass", fs=sampling_rate, output="sos"
    )


def channel_map(traces: list[Trace]) -> dict[tuple[NSL, str], Trace]:
    """Map station and orientation code to the trace."""
    from qseek.utils import NSL

    return {(NSL(*tr.nslc_id[:3]), tr.channel[-1:]): tr for tr in traces}


def correlate_window(
    pairs: list[tuple[Trace, Trace]],
    start_1: float,
    start_2: float,
    duration: float,
    max_lag: float,
) -> tuple[float, float, float] | None:
    """Correlate the window of event 1 against the longer window of event 2.

    The normalized correlations of all components are stacked: the components
    form one vector, which does not depend on the orientation of the horizontals.

    Args:
        pairs: Traces of the components of event 1 and event 2, with the same
            sampling interval.
        start_1: Start of the template window of event 1, timestamp.
        start_2: Start of the search window of event 2, timestamp. The search window
            is `2 * max_lag` longer than the template.
        duration: Duration of the template window in s.
        max_lag: Maximum lag in s.

    Returns:
        tuple[float, float, float] | None: The start time of the template, the
            matched start time in event 2 with sub-sample precision, and the
            correlation coefficient. `None` if the windows exceed the traces, the
            data are constant or the maximum lies at the maximum lag.
    """
    deltat = pairs[0][0].deltat
    n_samples = round(duration / deltat)
    n_lags = 2 * round(max_lag / deltat) + 1
    if n_samples < MIN_SAMPLES:
        return None

    tr_1, tr_2 = pairs[0]
    idx_1 = round((start_1 - tr_1.tmin) / deltat)
    idx_2 = round((start_2 - tr_2.tmin) / deltat)
    templates, searches = [], []
    for tr_a, tr_b in pairs:
        # all components need the same sample times as the first one
        if abs(tr_a.tmin - tr_1.tmin) > 1e-3 * deltat:
            continue
        if abs(tr_b.tmin - tr_2.tmin) > 1e-3 * deltat:
            continue
        if idx_1 < 0 or idx_2 < 0:
            continue
        template = tr_a.ydata[idx_1 : idx_1 + n_samples]
        search = tr_b.ydata[idx_2 : idx_2 + n_samples + n_lags - 1]
        if template.size != n_samples or search.size != n_samples + n_lags - 1:
            continue
        templates.append(template)
        searches.append(search)
    if not templates:
        return None

    coefficients = normalized_correlation(np.array(templates), np.array(searches))
    if coefficients is None:
        return None
    peak = int(np.argmax(coefficients))
    if peak == 0 or peak == n_lags - 1:
        return None
    offset, coefficient = parabolic_peak(coefficients, peak)
    matched_1 = tr_1.tmin + idx_1 * deltat
    matched_2 = tr_2.tmin + (idx_2 + peak + offset) * deltat
    return matched_1, matched_2, min(coefficient, 1.0)


def normalized_correlation(
    templates: np.ndarray, searches: np.ndarray
) -> np.ndarray | None:
    """Normalized cross-correlation of templates sliding along longer windows.

    Args:
        templates: Template windows, shape (components, samples).
        searches: Search windows, shape (components, samples + lags - 1).

    Returns:
        np.ndarray | None: The correlation coefficient at each lag, stacked over the
            components, or `None` if the data are constant.
    """
    templates = templates.astype(np.float64)
    searches = searches.astype(np.float64)
    n_samples = templates.shape[1]
    templates = templates - templates.mean(axis=1, keepdims=True)

    numerator = sum(
        np.correlate(search, template, mode="valid")
        for template, search in zip(templates, searches, strict=True)
    )
    padded = np.pad(searches, ((0, 0), (1, 0)))
    sums = np.cumsum(padded, axis=1)
    squares = np.cumsum(padded**2, axis=1)
    window_sums = sums[:, n_samples:] - sums[:, :-n_samples]
    window_squares = squares[:, n_samples:] - squares[:, :-n_samples]
    variance = np.sum(window_squares - window_sums**2 / n_samples, axis=0)
    energy = np.sum(templates**2)
    max_variance = variance.max()
    if energy <= 0.0 or max_variance <= 0.0:
        return None
    # rounding errors of the cumulative sums, e.g. for zeros in the search window
    variance = np.maximum(variance, 1e-9 * max_variance)
    return numerator / np.sqrt(energy * variance)


def parabolic_peak(values: np.ndarray, peak: int) -> tuple[float, float]:
    """Sub-sample offset and value of a maximum from a parabola through 3 samples."""
    left, center, right = values[peak - 1], values[peak], values[peak + 1]
    curvature = left - 2.0 * center + right
    if curvature >= 0.0:
        return 0.0, float(center)
    offset = 0.5 * (left - right) / curvature
    return float(offset), float(center - 0.25 * (left - right) * offset)
