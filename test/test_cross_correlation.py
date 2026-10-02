from __future__ import annotations

import asyncio
from collections import Counter
from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import Field, ValidationError
from pyrocko import trace

from qseek.exporters.cross_correlation import (
    CorrelationEvent,
    CrossCorrelation,
    PhaseWindow,
    WaveformCache,
    correlate_window,
    normalized_correlation,
)
from qseek.models.location import Location
from qseek.utils import NSL

STATION = NSL("XX", "STA", "")

DELTAT = 0.01


def wavelet(times: np.ndarray) -> np.ndarray:
    return np.exp(-((times / 0.1) ** 2)) * (
        np.sin(2 * np.pi * 6.0 * times) + 0.5 * np.sin(2 * np.pi * 9.5 * times + 1.0)
    )


def make_trace(arrival: float, tmin: float, channel: str = "HHZ") -> trace.Trace:
    times = tmin + np.arange(400) * DELTAT
    return trace.Trace(
        "XX",
        "STA",
        "",
        channel,
        tmin=tmin,
        deltat=DELTAT,
        ydata=wavelet(times - arrival),
    )


@pytest.mark.parametrize("shift", [0.0, 0.0123, -0.0377, 0.081])
def test_correlate_window(shift: float) -> None:
    # the arrival of event 2 is shifted by `shift` against its window anchor
    arrival_1, arrival_2 = 1.502, 2.234
    tr_1 = make_trace(arrival_1, tmin=0.0)
    tr_2 = make_trace(arrival_2 + shift, tmin=0.0037)
    result = correlate_window(
        [(tr_1, tr_2)],
        start_1=arrival_1 - 0.2,
        start_2=arrival_2 - 0.2 - 0.15,
        duration=0.6,
        max_lag=0.15,
    )
    assert result is not None
    matched_1, matched_2, coefficient = result
    assert coefficient > 0.99
    # the matched window starts have the offset of the arrivals
    assert (matched_2 - matched_1) == pytest.approx(
        arrival_2 + shift - arrival_1, abs=5e-4
    )


def test_correlate_window_max_lag() -> None:
    tr_1 = make_trace(1.5, tmin=0.0)
    tr_2 = make_trace(1.8, tmin=0.0)
    result = correlate_window(
        [(tr_1, tr_2)], start_1=1.3, start_2=1.3 - 0.1, duration=0.6, max_lag=0.1
    )
    assert result is None


def test_normalized_correlation() -> None:
    rng = np.random.default_rng(0)
    search = rng.normal(size=(2, 120))
    template = 3.0 * search[:, 30:80] + 1.0
    coefficients = normalized_correlation(template, search)
    assert coefficients is not None
    assert coefficients.size == 71
    assert int(np.argmax(coefficients)) == 30
    assert coefficients[30] == pytest.approx(1.0)
    assert np.all(np.abs(coefficients) <= 1.0 + 1e-9)
    assert normalized_correlation(np.ones((1, 50)), search[:1]) is None


def test_select_pairs() -> None:
    events = [
        CorrelationEvent(
            id=i + 1,
            detection=Location(lat=40.0, lon=14.0, east_shift=i * 1000.0, depth=2000.0),
            origin=0.0,
            arrivals={},
        )
        for i in range(4)
    ]
    settings = CrossCorrelation(max_separation=1500.0)
    assert settings.select_pairs(events) == [(0, 1), (1, 2), (2, 3)]
    settings = CrossCorrelation(max_separation=2500.0, max_neighbors=1)
    assert settings.select_pairs(events) == [(0, 1), (1, 2), (2, 3)]
    settings = CrossCorrelation(max_separation=2500.0)
    assert (0, 2) in settings.select_pairs(events)
    assert settings.select_pairs(events[:1]) == []


def test_waveform_cache() -> None:
    def traces(n_samples: int) -> list[trace.Trace]:
        return [trace.Trace(ydata=np.zeros(n_samples, dtype=np.float32), deltat=DELTAT)]

    cache = WaveformCache(max_bytes=1000)
    cache["a"] = traces(100)
    cache["b"] = traces(100)
    assert cache.get("a") is not None  # a is now the most recent
    cache["c"] = traces(100)
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.n_bytes == 800
    cache["d"] = traces(1000)  # larger than the cache, kept alone
    assert cache.get("d") is not None
    assert cache.get("a") is None
    assert cache.n_bytes == 4000


def test_waveform_cache_same_key() -> None:
    cache = WaveformCache(max_bytes=10_000)
    cache["a"] = [trace.Trace(ydata=np.zeros(100, dtype=np.float32))]
    cache["a"] = [trace.Trace(ydata=np.zeros(100, dtype=np.float32))]
    assert cache.n_bytes == 400


def test_phase_window_components() -> None:
    with pytest.raises(ValidationError):
        PhaseWindow(seconds_before=0.1, seconds_after=0.5, max_lag=0.2, components="ZZ")


def noise_trace(
    channel: str = "HHZ",
    tmin: float = 0.0,
    duration: float = 20.0,
    deltat: float = DELTAT,
    station: str = "STA",
) -> trace.Trace:
    rng = np.random.default_rng(0)
    n_samples = round(duration / deltat)
    return trace.Trace(
        "XX",
        station,
        "",
        channel,
        tmin=tmin,
        deltat=deltat,
        ydata=rng.normal(size=n_samples),
    )


def test_filter_waveforms() -> None:
    settings = CrossCorrelation()  # 3 s padding for the 1 Hz low corner
    spans = {STATION: (9.0, 11.0)}
    traces = [
        noise_trace("HHZ"),
        noise_trace("HHN", tmin=8.0),  # the padding is not covered
        noise_trace("HHE", deltat=0.6),  # Nyquist frequency below 1 Hz
        noise_trace("HHZ", station="OTHER"),  # not requested
        noise_trace("HHX"),  # component not correlated
    ]
    stats: Counter[str] = Counter()
    filtered = settings.filter_waveforms(traces, spans, stats)
    assert [tr.channel for tr in filtered] == ["HHZ"]
    assert stats == {"filtered": 1, "no_data": 1, "nyquist": 1}
    tr = filtered[0]
    assert tr.tmin == pytest.approx(9.0, abs=3 * DELTAT)
    assert tr.tmax == pytest.approx(11.0, abs=3 * DELTAT)
    assert tr.ydata.dtype == np.float32
    # the sample times stay on the grid of the input trace
    assert (tr.tmin / DELTAT) == pytest.approx(round(tr.tmin / DELTAT))


def correlation_event(id: int, arrivals: dict[str, float]) -> CorrelationEvent:
    return CorrelationEvent(
        id=id,
        detection=Location(lat=40.0, lon=14.0),
        origin=0.0,
        arrivals={(STATION, phase): time for phase, time in arrivals.items()},
    )


def test_correlate_pair() -> None:
    settings = CrossCorrelation()
    arrival_1, arrival_2 = 1.502, 2.234
    traces_1 = [make_trace(arrival_1, tmin=0.0)]
    traces_2 = [make_trace(arrival_2, tmin=0.0)]
    event_1 = correlation_event(1, {"P": arrival_1})
    event_2 = correlation_event(2, {"P": arrival_2})
    (time,) = settings.correlate_pair(event_1, event_2, traces_1, traces_2)
    assert time.nsl == STATION
    assert time.phase == "P"
    assert time.coefficient > 0.99
    assert time.time == pytest.approx(arrival_1 - arrival_2, abs=5e-4)

    # the S window of event 1 starts 0.1 s after P: too little P window is left
    close_s = correlation_event(1, {"P": arrival_1, "S": arrival_1 + 0.3})
    assert settings.correlate_pair(close_s, event_2, traces_1, traces_2) == []
    # a later S keeps enough of the P window
    later_s = correlation_event(1, {"P": arrival_1, "S": arrival_1 + 0.5})
    assert len(settings.correlate_pair(later_s, event_2, traces_1, traces_2)) == 1

    # other channel or sampling interval at event 2
    other_channel = [make_trace(arrival_2, tmin=0.0, channel="EHZ")]
    assert settings.correlate_pair(event_1, event_2, traces_1, other_channel) == []
    resampled = make_trace(arrival_2, tmin=0.0)
    resampled.deltat = 0.02
    assert settings.correlate_pair(event_1, event_2, traces_1, [resampled]) == []


class Detection(Location):
    uid: UUID = Field(default_factory=uuid4)


def test_correlate_loads_each_event_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[int] = Counter()

    async def load_waveforms(self, event, waveform_provider, stats=None):
        calls[event.id] += 1
        await asyncio.sleep(0.01)
        stats["filtered"] += 1
        return [trace.Trace(ydata=np.zeros(10, dtype=np.float32))]

    monkeypatch.setattr(CrossCorrelation, "load_waveforms", load_waveforms)
    monkeypatch.setattr(CrossCorrelation, "correlate_pair", lambda self, *args: [])
    events = [
        CorrelationEvent(
            id=i + 1,
            detection=Detection(lat=40.0, lon=14.0, east_shift=10.0 * i),
            origin=0.0,
            arrivals={},
        )
        for i in range(12)
    ]
    settings = CrossCorrelation(max_neighbors=5, n_parallel=4)
    results = asyncio.run(settings.correlate(events, waveform_provider=None))
    assert results == {}
    # the events wait for the loads of their neighbors, each event loads once
    assert set(calls) == {ev.id for ev in events}
    assert set(calls.values()) == {1}
