from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from obspy import Stream
from pyrocko.trace import Trace

from qseek.images.batch_size import BatchSizeTuner
from qseek.images.seisbench import SeisBench

N_WINDOWS = 10_000


def run(tuner: BatchSizeTuner, rates: dict[int, float], n_calls: int) -> list[int]:
    """Run the tuner against fixed throughputs in windows per second."""
    batch_sizes = []
    for _ in range(n_calls):
        batch_size = tuner.next_batch_size()
        batch_sizes.append(batch_size)
        tuner.record(batch_size, N_WINDOWS, N_WINDOWS / rates[batch_size])
    return batch_sizes


def peaked_rates(best: int) -> dict[int, float]:
    """Throughput that falls by 20% per factor of two away from `best`."""
    sizes = [2**exp for exp in range(5, 13)]
    return {size: 1000.0 * 0.8 ** abs(np.log2(size / best)) for size in sizes}


@pytest.mark.parametrize("best", [32, 64, 256, 1024, 4096])
def test_finds_fastest_batch_size(best: int):
    tuner = BatchSizeTuner()
    batch_sizes = run(tuner, peaked_rates(best), n_calls=80)
    assert tuner.batch_size == best
    assert tuner._probe is None
    assert all(32 <= size <= 4096 for size in batch_sizes)


def test_keeps_batch_size_on_flat_throughput():
    rng = random.Random(0)
    tuner = BatchSizeTuner()
    for _ in range(500):
        batch_size = tuner.next_batch_size()
        rate = 1000.0 * rng.uniform(0.98, 1.02)
        tuner.record(batch_size, N_WINDOWS, N_WINDOWS / rate)
    assert tuner.batch_size == 256


def test_first_call_is_not_measured():
    tuner = BatchSizeTuner()
    rates = peaked_rates(1024)
    for _ in range(80):
        batch_size = tuner.next_batch_size()
        # The first call of every batch size is slow, e.g. due to compilation
        warm = batch_size in tuner._warm
        seconds = N_WINDOWS / rates[batch_size] * (1.0 if warm else 50.0)
        tuner.record(batch_size, N_WINDOWS, seconds)
    assert tuner.batch_size == 1024


def test_probes_again_after_interval():
    tuner = BatchSizeTuner(probe_interval=10)
    run(tuner, peaked_rates(256), n_calls=40)
    assert tuner.batch_size == 256
    # The load on the machine changes
    run(tuner, peaked_rates(1024), n_calls=60)
    assert tuner.batch_size == 1024


def test_ignores_empty_calls():
    tuner = BatchSizeTuner()
    for _ in range(20):
        tuner.record(tuner.next_batch_size(), 0, 1.0)
        tuner.record(tuner.next_batch_size(), 100, 0.0)
    assert tuner.batch_size == 256
    assert not tuner._warm


def test_out_of_memory_limits_batch_size():
    tuner = BatchSizeTuner()
    rates = peaked_rates(4096)
    for _ in range(80):
        batch_size = tuner.next_batch_size()
        if batch_size > 512:
            assert tuner.record_out_of_memory(batch_size)
            continue
        tuner.record(batch_size, N_WINDOWS, N_WINDOWS / rates[batch_size])
    assert tuner.batch_size == 512
    assert tuner.max_batch_size == 512

    assert not BatchSizeTuner(batch_size=32).record_out_of_memory(32)


def make_traces(n_stations: int, n_samples: int) -> list[Trace]:
    return [
        Trace(
            network="XX",
            station=f"S{station:02d}",
            channel=f"HH{component}",
            tmin=1700000000.0,
            deltat=0.01,
            ydata=np.zeros(n_samples, dtype=np.float32),
        )
        for station in range(n_stations)
        for component in "ZNE"
    ]


def test_count_windows():
    function = SeisBench(window_overlap_samples=2000)
    function._seisbench_model = SimpleNamespace(in_samples=3001, sampling_rate=100.0)
    # Starts at 0, 1001, ..., 6006 and a last window at the end of the trace
    assert function._count_windows(make_traces(3, 10_000)) == 3 * 8
    assert function._count_windows(make_traces(2, 3001)) == 2
    assert function._count_windows(make_traces(2, 3000)) == 0

    # SeisBench resamples to the sampling rate of the model
    function._seisbench_model.sampling_rate = 200.0
    assert function._count_windows(make_traces(1, 5000)) == 8


@pytest.mark.asyncio
async def test_retry_with_smaller_batch_size_on_out_of_memory():
    batch_sizes: list[int] = []

    async def annotate(stream: Stream, batch_size: int, **kwargs) -> Stream:
        batch_sizes.append(batch_size)
        if batch_size > 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        annotations = Stream()
        for trace in stream:
            for phase in "PS":
                annotation = trace.copy()
                annotation.stats.channel = f"PhaseNet_{phase}"
                annotations.append(annotation)
        return annotations

    function = SeisBench(sampling_rate=100.0)
    function._seisbench_model = SimpleNamespace(
        annotate_async=annotate, sampling_rate=100.0, in_samples=3001
    )

    images = await function.process_traces(make_traces(2, 10_000))
    assert batch_sizes == [256, 128, 64]
    assert all(image.n_traces == 6 for image in images)
    assert function._batch_size_tuner.max_batch_size == 64

    # A fixed batch size is not retried
    function = SeisBench(sampling_rate=100.0, batch_size=128)
    function._seisbench_model = SimpleNamespace(
        annotate_async=annotate, sampling_rate=100.0, in_samples=3001
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        await function.process_traces(make_traces(1, 10_000))
