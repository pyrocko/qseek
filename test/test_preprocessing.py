from datetime import datetime, timezone

import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.pre_processing.base import BatchPreProcessing, split_traces
from qseek.pre_processing.frequency_filters import Bandpass, Highpass, Lowpass
from qseek.pre_processing.resample import Downsample, Resample, downsample, resample
from qseek.waveforms.base import WaveformBatch


@pytest.fixture
def traces(n_traces: int = 100, n_samples: int = 10000):
    rng = np.random.default_rng(0)
    traces = []
    for itr in range(n_traces):
        tr = Trace(
            network="XX",
            station=f"ST{itr:03d}",
            location="",
            channel="BHZ",
            deltat=0.01,
            tmin=0.0,
            ydata=rng.standard_normal(n_samples),
        )
        traces.append(tr)
    return traces


def test_resampling(traces):
    delta_t = 0.01  # no resampling
    resampled_traces = resample(traces, delta_t=delta_t, demean=False)

    for tr, tr_resampled in zip(traces, resampled_traces, strict=True):
        assert tr_resampled.deltat == delta_t
        np.testing.assert_allclose(tr.ydata, tr_resampled.ydata)


@pytest.mark.parametrize(
    "method,delta_t",
    [(resample, delta_t) for delta_t in (0.005, 0.02, 0.04, 0.05, 0.1)]
    + [(downsample, delta_t) for delta_t in (0.02, 0.04, 0.05, 0.1)],
)
def test_resampling_sine(method, delta_t: float):
    frequency = 2.0
    times = np.arange(10000) * 0.01
    traces = [
        Trace(
            network="XX",
            station=f"ST{itr:03d}",
            channel="BHZ",
            tmin=0.0,
            deltat=0.01,
            ydata=np.sin(2 * np.pi * frequency * times + itr),
        )
        for itr in range(3)
    ]

    n_samples = round(100.0 / delta_t)
    for itr, tr in enumerate(method(traces, delta_t=delta_t)):
        assert tr.deltat == delta_t
        # Within half an input sample, downsample compensates the FIR delay
        assert abs(tr.tmin) <= 0.005
        if method is resample:
            assert tr.ydata.size == n_samples
        else:
            # The FIR filter delay is compensated by dropping samples at the end
            assert 0.95 * n_samples < tr.ydata.size <= n_samples
        # Compare with the analytic signal, excluding the edges
        interior = slice(tr.ydata.size // 10, -tr.ydata.size // 10)
        expected = np.sin(2 * np.pi * frequency * tr.get_xdata() + itr)
        np.testing.assert_allclose(tr.ydata[interior], expected[interior], atol=1e-3)


@pytest.mark.benchmark(group="resampling")
@pytest.mark.parametrize("method", ["downsample", "resample"])
def test_resampling_benchmark(benchmark, traces, method: str):
    if method == "downsample":
        func = downsample
    elif method == "resample":
        func = resample
    else:
        raise ValueError(f"Unknown method: {method}")

    benchmark(func, traces, delta_t=0.04, demean=True)


def _mixed_batch() -> WaveformBatch:
    """Stations at 200 and 250 Hz, with lengths differing by one sample."""
    rng = np.random.default_rng(1)
    traces = []
    for ista in range(20):
        sampling_rate = 200.0 if ista < 15 else 250.0
        n_samples = int(10 * sampling_rate) + ista % 2
        for channel in ("HHE", "HHN", "HHZ"):
            traces.append(
                Trace(
                    network="XX",
                    station=f"ST{ista:03d}",
                    channel=channel,
                    tmin=0.0,
                    deltat=1.0 / sampling_rate,
                    ydata=rng.integers(-1000, 1000, n_samples).astype(np.int32),
                )
            )
    now = datetime.now(tz=timezone.utc)
    return WaveformBatch(traces=traces, start_time=now, end_time=now, i_batch=0)


def test_split_traces():
    traces = list(range(10))
    assert split_traces([], 4) == []
    assert split_traces(traces, 1) == [traces]
    assert split_traces(traces, 4) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert split_traces(traces, 20) == [[t] for t in traces]


def test_filter_traces():
    batch = _mixed_batch()
    module = Resample(stations=["XX.ST001", "XX.ST01*", "YY.ST002"])

    selected = module.filter_traces(batch)
    stations = sorted({tr.station for tr in selected})
    assert stations == ["ST001"] + [f"ST{i:03d}" for i in range(10, 20)]
    assert len(selected) == 3 * len(stations)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [
        Resample(sampling_frequency=100.0),
        Downsample(sampling_frequency=50.0),
        Bandpass(bandpass=(0.5, 30.0)),
        Highpass(frequency=1.0),
        Lowpass(frequency=20.0),
    ],
)
async def test_chunks_identical(module: BatchPreProcessing):
    single = module.model_copy(update={"n_threads": 1})
    reference, chunked = _mixed_batch(), _mixed_batch()

    await single.process_batch(reference)
    await module.process_batch(chunked)

    assert module.n_threads > 1
    for tr_ref, tr in zip(reference.traces, chunked.traces, strict=True):
        assert tr.deltat == tr_ref.deltat
        assert tr.tmin == tr_ref.tmin
        np.testing.assert_array_equal(tr.ydata, tr_ref.ydata)
