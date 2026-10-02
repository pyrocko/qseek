from __future__ import annotations

import numpy as np
import pytest
from pyrocko import trace

from qseek.exporters.cross_correlation import (
    CorrelationEvent,
    CrossCorrelation,
    WaveformCache,
    correlate_window,
    normalized_correlation,
)
from qseek.models.location import Location

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
