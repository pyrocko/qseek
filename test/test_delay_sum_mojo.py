"""Parity tests for the Mojo delay_sum (src/qseek/ext_mojo/delay_sum.mojo).

Compares the Mojo implementation against both the reference `pyrocko.parstack`
and against qseek's own compiled C extension (`qseek.ext.delay_sum`), which is
the function this Mojo module is a port of.

All three Mojo entry points take two Python lists -- `TraceInput` and
`NodeStack` (see `qseek.reduce`) -- where the C extension and pyrocko take
flat arrays: traces plus a parallel `offsets` array, and packed 2-D
(shifts, weights) plus a `node_mask`. `to_trace_inputs()` and
`to_node_stacks()` below convert one to the other, so every test drives all
implementations from the same generated data.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Literal, get_args

import numpy as np
import pytest
from pyrocko import parstack as pyrocko_parstack

pytest.importorskip(
    "mojo.importer", reason="Mojo toolchain (dependency 'modular') not installed"
)

from qseek.ext import array_tools
from qseek.ext import delay_sum as qseek_delay_sum
from qseek.ext_mojo import delay_sum as mojo_delay_sum
from qseek.reduce import NodeStack, TraceInput

N_THREADS_TEST = [1, 2]


def get_data(n_nodes: int = 20, n_samples: int = 4_000, n_traces: int = 10):
    rng = np.random.default_rng(123)

    traces = []
    for _ in range(n_traces):
        traces.append(rng.uniform(0, 1.0, (n_samples)).astype(np.float32))

    offsets = rng.integers(0, 1, size=n_traces, dtype=np.int32)
    shifts = rng.integers(-200, -5, size=(n_nodes, n_traces)).astype(np.int32)
    weights = rng.uniform(0, 1.0, size=(n_nodes, n_traces)).astype(np.float32)

    return traces, offsets, shifts, weights


def to_trace_inputs(traces: list[np.ndarray], offsets: np.ndarray) -> list[TraceInput]:
    """Pair each trace array with its offset, as the Mojo entry points take."""
    return [
        TraceInput(data=data, offset=int(offset))
        for data, offset in zip(traces, offsets, strict=True)
    ]


def to_node_stacks(
    shifts: np.ndarray, weights: np.ndarray, indices: list[int] | None = None
) -> list[NodeStack]:
    """Build the node list the Mojo entry points take, from flat arrays.

    `indices` defaults to each row's own position, but callers simulating a
    masked/incremental subset (like a real `DelaySumReduce.stack()` call)
    pass the subset's *global* indices explicitly.
    """
    if indices is None:
        indices = range(shifts.shape[0])
    return [
        NodeStack(index=i, shifts=shifts[row], weights=weights[row])
        for row, i in enumerate(indices)
    ]


@pytest.fixture
def data():
    return get_data()


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_mojo_delay_sum_matches_pyrocko_and_qseek(data, n_threads: int):
    traces, offsets, shifts, weights = data

    mojo_stack, mojo_offset = mojo_delay_sum.delay_sum(
        to_trace_inputs(traces, offsets),
        to_node_stacks(shifts, weights),
        n_threads=n_threads,
    )
    qseek_stack, qseek_offset = qseek_delay_sum.delay_sum(
        traces, offsets, shifts, weights, n_threads=n_threads
    )
    pyrocko_stack, pyrocko_offset = pyrocko_parstack.parstack(
        traces,
        offsets,
        shifts,
        weights,
        method=0,
        nparallel=n_threads,
        dtype=np.float32,
    )

    assert mojo_offset == qseek_offset == pyrocko_offset
    np.testing.assert_allclose(mojo_stack, qseek_stack, rtol=1e-5)
    np.testing.assert_allclose(mojo_stack, pyrocko_stack, rtol=1e-5)


@pytest.mark.parametrize("length", [500, 501, 504])
def test_mojo_delay_sum_shift_range_and_stack_reuse(data, length: int):
    # `length` is swept across `length % 8` because the C extension's SIMD
    # loop used to run up to 7 lanes past the end whenever an explicit
    # shift_range put the window above a trace's shifted start and the
    # window length was not a multiple of the lane width -- writing past the
    # node's stack row. It stayed in bounds for length % 8 == 0, which is why
    # the older lengthout=1000 test never caught it. Fixed in delay_sum.c;
    # this keeps all three implementations pinned together on that path.
    traces, offsets, shifts, weights = data

    shift_range = (0, length)
    trace_inputs = to_trace_inputs(traces, offsets)
    nodes = to_node_stacks(shifts, weights)
    res = None
    qseek_res = None
    pyrocko_res = None
    for _ in range(3):
        res, offset = mojo_delay_sum.delay_sum(
            trace_inputs, nodes, shift_range=shift_range, stack=res
        )
        qseek_res, qseek_offset = qseek_delay_sum.delay_sum(
            traces, offsets, shifts, weights, shift_range=shift_range, stack=qseek_res
        )
        pyrocko_res, pyrocko_offset = pyrocko_parstack.parstack(
            traces,
            offsets,
            shifts,
            weights,
            method=0,
            result=pyrocko_res,
            lengthout=length,
            dtype=np.float32,
        )

    assert offset == qseek_offset == pyrocko_offset == 0
    assert res.shape == (shifts.shape[0], length)
    np.testing.assert_allclose(res, qseek_res, rtol=1e-5)
    np.testing.assert_allclose(res, pyrocko_res, rtol=1e-5)


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_mojo_delay_sum_reduce_matches(data, n_threads: int):
    traces, offsets, shifts, weights = data
    nodes = to_node_stacks(shifts, weights)

    mojo_max, mojo_idx, mojo_offset = mojo_delay_sum.delay_sum_reduce(
        to_trace_inputs(traces, offsets), nodes, n_threads=n_threads
    )
    qseek_max, qseek_idx, qseek_offset = qseek_delay_sum.delay_sum_reduce(
        traces, offsets, shifts, weights, n_threads=n_threads
    )

    res_pyrocko, offset_pyrocko = pyrocko_parstack.parstack(
        traces, offsets, shifts, weights, result=None, method=0, dtype=np.float32
    )
    pyrocko_idx, pyrocko_max = array_tools.argmax_masked(res_pyrocko)

    assert mojo_offset == qseek_offset == offset_pyrocko
    np.testing.assert_allclose(mojo_max, qseek_max, rtol=1e-5)
    np.testing.assert_allclose(mojo_max, pyrocko_max, rtol=1e-5)
    np.testing.assert_equal(mojo_idx, qseek_idx)
    np.testing.assert_equal(mojo_idx, pyrocko_idx)


def test_mojo_delay_sum_reduce_incremental_indices(data):
    """Splitting the node list across calls must still record global indices.

    Mirrors how `DelaySumReduce.stack()` only passes the not-yet-stacked
    subset each call: each `NodeStack.index` is the node's position in the
    *full* node list, not its position in this call's (sub)list.
    """
    traces, offsets, shifts, weights = data
    n_nodes = shifts.shape[0]
    split = n_nodes // 2

    shift_range = (0, 3_800)
    node_max = None
    node_max_idx = None
    for lo, hi in [(0, split), (split, n_nodes)]:
        nodes = to_node_stacks(
            shifts[lo:hi], weights[lo:hi], indices=list(range(lo, hi))
        )
        node_max, node_max_idx, offset = mojo_delay_sum.delay_sum_reduce(
            to_trace_inputs(traces, offsets),
            nodes,
            shift_range=shift_range,
            node_stack_max=node_max,
            node_stack_max_idx=node_max_idx,
        )

    qseek_max, qseek_idx, qseek_offset = qseek_delay_sum.delay_sum_reduce(
        traces, offsets, shifts, weights, shift_range=shift_range
    )

    assert offset == qseek_offset
    np.testing.assert_allclose(node_max, qseek_max, rtol=1e-5)
    np.testing.assert_equal(node_max_idx, qseek_idx)


def test_mojo_delay_sum_snapshot_matches(data):
    traces, offsets, shifts, weights = data
    nodes = to_node_stacks(shifts, weights)

    mojo_max, _, mojo_offset = mojo_delay_sum.delay_sum_reduce(
        to_trace_inputs(traces, offsets), nodes
    )
    _, _, qseek_offset = qseek_delay_sum.delay_sum_reduce(
        traces, offsets, shifts, weights
    )
    assert mojo_offset == qseek_offset

    for idx in random.Random(0).choices(range(len(mojo_max)), k=50):
        mojo_snap = mojo_delay_sum.delay_sum_snapshot(
            to_trace_inputs(traces, offsets), nodes, index=idx
        )
        qseek_snap = qseek_delay_sum.delay_sum_snapshot(
            traces, offsets, shifts, weights, index=idx
        )
        np.testing.assert_allclose(mojo_snap, qseek_snap, rtol=1e-5)
        np.testing.assert_allclose(mojo_snap.max(), mojo_max[idx], rtol=1e-5)


def test_mojo_delay_sum_reduce_node_subset(data):
    """Excluding nodes from the list must match the C extension's node_mask.

    Mojo has no `node_mask` concept for `delay_sum_reduce`/`delay_sum_snapshot`
    -- callers (reduce.py) filter the node list themselves instead. This
    checks that filtering is equivalent to the C extension's node_mask.
    """
    traces, offsets, shifts, weights = data
    n_nodes = shifts.shape[0]

    mask = np.zeros(n_nodes, dtype=bool)
    masked_indices = random.Random(0).sample(range(n_nodes), 5)
    mask[masked_indices] = True

    kept_indices = [i for i in range(n_nodes) if not mask[i]]
    nodes = to_node_stacks(
        shifts[kept_indices], weights[kept_indices], indices=kept_indices
    )

    mojo_max, mojo_idx, mojo_offset = mojo_delay_sum.delay_sum_reduce(
        to_trace_inputs(traces, offsets), nodes
    )
    qseek_max, qseek_idx, qseek_offset = qseek_delay_sum.delay_sum_reduce(
        traces, offsets, shifts, weights, node_mask=mask
    )

    assert mojo_offset == qseek_offset
    np.testing.assert_allclose(mojo_max, qseek_max, rtol=1e-5)
    np.testing.assert_equal(mojo_idx, qseek_idx)
    assert not np.isin(mojo_idx, masked_indices).any()


# ===----------------------------------------------------------------=== #
# GIL release
# ===----------------------------------------------------------------=== #


def _spin_rate_during(work) -> float:
    """Python-bytecode throughput of another thread while `work` runs.

    A pure-Python counter loop can only advance while its thread holds the
    GIL, so its rate during `work` measures how much of the GIL `work`
    leaves available.
    """
    counter = 0
    running = True

    def spin() -> None:
        nonlocal counter
        while running:
            counter += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)  # let the spinner reach steady state
        start_count, start = counter, time.perf_counter()
        work()
        elapsed, spins = time.perf_counter() - start, counter - start_count
    finally:
        running = False
        thread.join()
    return spins / elapsed


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs >=2 cores to run spinner alongside"
)
def test_mojo_releases_gil(data):
    """The Mojo kernels must release the GIL while computing.

    `DelaySumReduce.stack()` offloads via `asyncio.to_thread`, which only
    buys concurrency if the extension actually drops the GIL -- the C
    extension this replaced did so with `Py_BEGIN_ALLOW_THREADS`. Without
    the `GILReleased` guard in delay_sum.mojo this ratio measures ~0.1.
    """
    traces, offsets, shifts, weights = get_data(
        n_nodes=400, n_samples=30_000, n_traces=100
    )
    trace_inputs = to_trace_inputs(traces, offsets)
    nodes = to_node_stacks(shifts, weights)

    # n_threads=1 so the kernel does not itself contend for every core.
    released = _spin_rate_during(lambda: time.sleep(0.3))
    during_mojo = _spin_rate_during(
        lambda: mojo_delay_sum.delay_sum_reduce(trace_inputs, nodes, n_threads=1)
    )

    fraction = during_mojo / released
    # Generous threshold: measures ~0.97 when released and ~0.10 when held,
    # so 0.5 tolerates a loaded CI machine without admitting a regression.
    assert fraction > 0.5, (
        f"Mojo kernel appears to hold the GIL "
        f"(spinner ran at {fraction:.2f} of its GIL-free rate)"
    )


# ===----------------------------------------------------------------=== #
# Benchmarks: pyrocko vs qseek (C) vs qseek (Mojo)
# ===----------------------------------------------------------------=== #
#
# Mirrors the shape of test/test_delay_sum.py's benchmarks, with "mojo"
# added as a third implementation.

N_THREADS_BENCH = [1, 2, 4]
ROUNDS = 4
Implementation = Literal["pyrocko", "qseek", "mojo"]


@pytest.fixture
def benchmark_data():
    return get_data(n_nodes=100, n_samples=30_000, n_traces=100)


@pytest.mark.benchmark(group="delay_sum")
@pytest.mark.parametrize("n_threads", N_THREADS_BENCH)
@pytest.mark.parametrize("implementation", get_args(Implementation))
def test_benchmark_delay_sum(
    benchmark,
    benchmark_data,
    n_threads: int,
    implementation: Implementation,
    rounds: int = ROUNDS,
):
    traces, offsets, shifts, weights = benchmark_data

    def stack_qseek() -> tuple[np.ndarray, int]:
        res = None
        for _ in range(rounds):
            res, offset = qseek_delay_sum.delay_sum(
                traces, offsets, shifts, weights, stack=res, n_threads=n_threads
            )
        array_tools.argmax_masked(res, n_threads=n_threads)
        return res, offset

    def stack_pyrocko() -> tuple[np.ndarray, int]:
        res = None
        for _ in range(rounds):
            res, offset = pyrocko_parstack.parstack(
                traces,
                offsets,
                shifts,
                weights,
                method=0,
                result=res,
                nparallel=n_threads,
                dtype=np.float32,
            )
        array_tools.argmax_masked(res, n_threads=n_threads)
        return res, offset

    def stack_mojo() -> tuple[np.ndarray, int]:
        trace_inputs = to_trace_inputs(traces, offsets)
        nodes = to_node_stacks(shifts, weights)
        res = None
        for _ in range(rounds):
            res, offset = mojo_delay_sum.delay_sum(
                trace_inputs, nodes, stack=res, n_threads=n_threads
            )
        array_tools.argmax_masked(res, n_threads=n_threads)
        return res, offset

    def benchmark_if(func, func_implementation: Implementation):
        return benchmark(func) if func_implementation == implementation else func()

    r_qseek, off_qseek = benchmark_if(stack_qseek, "qseek")
    r_pyrocko, off_pyrocko = benchmark_if(stack_pyrocko, "pyrocko")
    r_mojo, off_mojo = benchmark_if(stack_mojo, "mojo")

    np.testing.assert_allclose(r_pyrocko, r_qseek, rtol=1e-5)
    np.testing.assert_allclose(r_pyrocko, r_mojo, rtol=1e-5)
    assert off_pyrocko == off_qseek == off_mojo


@pytest.mark.benchmark(group="delay_sum_reduce")
@pytest.mark.parametrize("n_threads", N_THREADS_BENCH)
@pytest.mark.parametrize("implementation", get_args(Implementation))
def test_benchmark_delay_sum_reduce(
    benchmark, benchmark_data, n_threads: int, implementation: Implementation
):
    def stack_reduce_qseek():
        traces, offsets, shifts, weights = benchmark_data
        max_value, max_idx, offset = qseek_delay_sum.delay_sum_reduce(
            traces, offsets, shifts, weights, n_threads=n_threads
        )
        return max_idx, max_value, offset

    def stack_reduce_pyrocko():
        traces, offsets, shifts, weights = benchmark_data
        res, offset = pyrocko_parstack.parstack(
            traces,
            offsets,
            shifts,
            weights,
            result=None,
            method=0,
            nparallel=n_threads,
            dtype=np.float32,
        )
        max_idx, max_value = array_tools.argmax_masked(res, n_threads=n_threads)
        return max_idx, max_value, offset

    def stack_reduce_mojo():
        traces, offsets, shifts, weights = benchmark_data
        nodes = to_node_stacks(shifts, weights)
        max_value, max_idx, offset = mojo_delay_sum.delay_sum_reduce(
            to_trace_inputs(traces, offsets), nodes, n_threads=n_threads
        )
        return max_idx, max_value, offset

    def benchmark_if(func, func_implementation: Implementation):
        return benchmark(func) if func_implementation == implementation else func()

    qseek_max_idx, qseek_max, qseek_off = benchmark_if(stack_reduce_qseek, "qseek")
    pyr_max_idx, pyr_max, pyr_off = benchmark_if(stack_reduce_pyrocko, "pyrocko")
    mojo_max_idx, mojo_max, mojo_off = benchmark_if(stack_reduce_mojo, "mojo")

    np.testing.assert_allclose(pyr_max, qseek_max, rtol=1e-5)
    np.testing.assert_allclose(pyr_max, mojo_max, rtol=1e-5)
    np.testing.assert_equal(pyr_max_idx, qseek_max_idx)
    np.testing.assert_equal(pyr_max_idx, mojo_max_idx)
    assert pyr_off == qseek_off == mojo_off
