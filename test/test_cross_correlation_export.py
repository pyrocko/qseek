from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.exporters.cross_correlation.model import (
    EventCorrelationPair,
    _correlation_matrix,
)
from qseek.models.detection import EventDetection

SAMPLING_RATE = 100.0
DELTAT = 1.0 / SAMPLING_RATE


def new_event(time: datetime) -> EventDetection:
    return EventDetection(
        lat=10.0,
        lon=10.0,
        distance_border=1000.0,
        semblance=1.0,
        time=time,
    )


def sine_pulse(shift: float = 0.0, n_samples: int = 500) -> np.ndarray:
    t = np.arange(n_samples) * DELTAT
    return np.sin(2 * np.pi * 3.0 * (t - shift)) * np.exp(
        -((t - 2.0 - shift) ** 2) / 0.05
    )


def new_trace(station: str, ydata: np.ndarray, tmin: float = 0.0) -> Trace:
    return Trace(
        network="XX",
        station=station,
        location="",
        channel="Z",
        deltat=DELTAT,
        tmin=tmin,
        ydata=ydata.astype(np.float64),
    )


@pytest.mark.asyncio
async def test_calculate_correlation_matrix() -> None:
    """Stations common to both events get a diagonal-dominant coefficient matrix."""
    event_a = new_event(datetime(2020, 1, 1, tzinfo=timezone.utc))
    event_b = new_event(datetime(2020, 1, 1, 0, 0, 1, tzinfo=timezone.utc))

    rng = np.random.default_rng(42)
    noise_a = rng.standard_normal(500)
    noise_b = rng.standard_normal(500)

    traces_a = [
        new_trace("AAA", sine_pulse(shift=0.0)),
        new_trace("CCC", noise_a),
    ]
    traces_b = [
        new_trace("AAA", sine_pulse(shift=0.05)),
        new_trace("CCC", noise_b),
    ]

    result = await EventCorrelationPair.calculate(
        event_a=event_a,
        event_b=event_b,
        traces_a=traces_a,
        traces_b=traces_b,
        phase="P",
    )

    assert result.event_a is event_a
    assert result.event_b is event_b
    assert result.phase == "P"
    assert [nsl.station for nsl in result.stations] == ["AAA", "CCC"]
    assert result._cc_values.shape == (2, 2)

    # Coefficients are bounded by Cauchy-Schwarz normalization.
    assert np.all(np.abs(result._cc_values) <= 1.0 + 1e-5)

    # AAA is a near-identical, shifted waveform in both events: strong self-match.
    assert result._cc_values[0, 0] > 0.9
    # CCC is uncorrelated noise: no strong match anywhere in its row/column.
    assert result._cc_values[1, 1] < 0.5

    # AAA's waveform in event_b is delayed by 0.05 s relative to event_a.
    assert result._time_shift.shape == (2, 2)
    assert result._time_shift[0, 0] == pytest.approx(-0.05, abs=DELTAT)


@pytest.mark.asyncio
async def test_calculate_drops_stations_not_common_to_both_events() -> None:
    """A station recorded at only one of the two events is excluded from the result."""
    event_a = new_event(datetime(2020, 1, 1, tzinfo=timezone.utc))
    event_b = new_event(datetime(2020, 1, 1, 0, 0, 1, tzinfo=timezone.utc))

    traces_a = [
        new_trace("AAA", sine_pulse()),
        new_trace("BBB", sine_pulse()),
    ]
    traces_b = [new_trace("AAA", sine_pulse())]

    result = await EventCorrelationPair.calculate(
        event_a=event_a,
        event_b=event_b,
        traces_a=traces_a,
        traces_b=traces_b,
        phase="P",
    )

    assert [nsl.station for nsl in result.stations] == ["AAA"]
    assert result._cc_values.shape == (1, 1)


@pytest.mark.asyncio
async def test_calculate_raises_without_common_stations() -> None:
    event_a = new_event(datetime(2020, 1, 1, tzinfo=timezone.utc))
    event_b = new_event(datetime(2020, 1, 1, 0, 0, 1, tzinfo=timezone.utc))

    traces_a = [new_trace("AAA", sine_pulse())]
    traces_b = [new_trace("BBB", sine_pulse())]

    with pytest.raises(ValueError, match="No common stations"):
        await EventCorrelationPair.calculate(
            event_a=event_a,
            event_b=event_b,
            traces_a=traces_a,
            traces_b=traces_b,
            phase="P",
        )


@pytest.mark.asyncio
async def test_calculate_resamples_to_coarsest_sampling_rate() -> None:
    """Stations with differing sampling rates are still comparable."""
    event_a = new_event(datetime(2020, 1, 1, tzinfo=timezone.utc))
    event_b = new_event(datetime(2020, 1, 1, 0, 0, 1, tzinfo=timezone.utc))

    trace_a = new_trace("AAA", sine_pulse())
    trace_b = new_trace("AAA", sine_pulse(shift=0.05))
    trace_b.deltat = DELTAT / 2
    trace_b.set_ydata(
        np.interp(
            np.arange(trace_a.ydata.size * 2) * trace_b.deltat,
            np.arange(trace_a.ydata.size) * trace_a.deltat,
            trace_b.ydata[: trace_a.ydata.size],
        )
    )

    result = await EventCorrelationPair.calculate(
        event_a=event_a,
        event_b=event_b,
        traces_a=[trace_a],
        traces_b=[trace_b],
        phase="P",
    )

    assert result._cc_values[0, 0] > 0.9
    assert result._time_shift[0, 0] == pytest.approx(-0.05, abs=DELTAT)


def test_origin_time_differences() -> None:
    event_a = new_event(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    event_b = new_event(datetime(2020, 1, 1, 0, 0, 5, tzinfo=timezone.utc))

    cc = EventCorrelationPair(
        event_a=event_a,
        event_b=event_b,
        stations=[],
        phase="P",
    )
    assert cc.origin_time_differences == timedelta(seconds=5)


def test_correlation_matrix_matches_reference_correlation() -> None:
    """The vectorized FFT batch correlation matches a direct numpy reference."""
    rng = np.random.default_rng(7)
    data_a = rng.standard_normal(37).astype(np.float32)
    data_b = rng.standard_normal(53).astype(np.float32)

    trace_a = new_trace("AAA", data_a)
    trace_b = new_trace("AAA", data_b)

    time_shift, cc_values = _correlation_matrix([trace_a], [trace_b])

    reference = np.correlate(data_a, data_b, mode="full")
    ref_index = int(np.argmax(reference))
    ref_lag_samples = ref_index - (data_b.size - 1)
    expected_cc = reference.max() / np.sqrt(np.sum(data_a**2) * np.sum(data_b**2))

    assert cc_values.shape == (1, 1)
    assert time_shift.shape == (1, 1)
    np.testing.assert_allclose(cc_values[0, 0], expected_cc, rtol=1e-4)
    np.testing.assert_allclose(time_shift[0, 0], ref_lag_samples * DELTAT, rtol=1e-4)


def test_correlation_matrix_respects_tmin_offset() -> None:
    """The time shift accounts for each trace's absolute tmin, not just its sample lag."""

    def gaussian_pulse(
        n_samples: int, center_index: int, width: float = 1.5
    ) -> np.ndarray:
        x = np.arange(n_samples)
        return np.exp(-0.5 * ((x - center_index) / width) ** 2)

    trace_a = new_trace("AAA", gaussian_pulse(20, center_index=10), tmin=100.0)
    trace_b = new_trace("AAA", gaussian_pulse(20, center_index=12), tmin=100.5)

    true_time_a = trace_a.tmin + 10 * DELTAT
    true_time_b = trace_b.tmin + 12 * DELTAT
    expected_time_shift = true_time_a - true_time_b

    time_shift, cc_values = _correlation_matrix([trace_a], [trace_b])

    assert cc_values[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert time_shift[0, 0] == pytest.approx(expected_time_shift, abs=1e-4)
