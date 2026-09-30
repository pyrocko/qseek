import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.pre_processing.resample import downsample, resample


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
