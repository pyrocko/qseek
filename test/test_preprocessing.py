import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.pre_processing.resample import downsample, resample


@pytest.fixture
def traces(n_traces: int = 100, n_samples: int = 10000):
    traces = []
    for itr in range(n_traces):
        tr = Trace(
            network="XX",
            station=f"ST{itr:03d}",
            location="",
            channel="BHZ",
            deltat=0.01,
            tmin=0.0,
            ydata=np.random.randn(n_samples),
        )
        traces.append(tr)
    return traces


def test_resampling(traces):
    delta_t = 0.01  # no resampling
    resampled_traces = resample(traces, delta_t=delta_t, demean=False)

    for tr, tr_resampled in zip(traces, resampled_traces, strict=True):
        assert tr.deltat == delta_t
        np.testing.assert_allclose(tr.ydata, tr_resampled.ydata)

    for delta_t in (0.005, 0.02, 0.05, 0.1):
        pyrocko_resampled = [tr.copy() for tr in traces]
        for tr in pyrocko_resampled:
            tr.resample(deltat=delta_t)
        resampled_traces = resample(traces, delta_t=delta_t, demean=True)
        for _tr_py, tr_qs in zip(pyrocko_resampled, resampled_traces, strict=True):
            assert tr_qs.deltat == delta_t

        # snuffle(pyrocko_resampled + resampled_traces)


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
