from __future__ import annotations

import random
from typing import Callable, Literal, get_args

import numpy as np
import pytest
from pyrocko import parstack as pyrocko_parstack
from pytest import fixture

from qseek.ext import array_tools, delay_sum

N_THREADS_TEST = [1, 2, 4]  # GHA Runners have 4 cores
ROUNDS = 4
Implementation = Literal["pyrocko", "qseek"]


def get_data(n_nodes: int = 100, n_samples: int = 30_000, n_traces: int = 100):
    # n_nodes = 50
    # n_samples = 30_000
    # n_traces = 4

    rng = np.random.default_rng(123)

    traces = []
    for _ in range(n_traces):
        traces.append(rng.uniform(0, 1.0, (n_samples)).astype(np.float32))

    offsets = rng.integers(0, 1, size=n_traces, dtype=np.int32)
    shifts = rng.integers(-2000, -50, size=(n_nodes, n_traces)).astype(np.int32)
    weights = rng.uniform(0, 1.0, size=(n_nodes, n_traces)).astype(np.float32)

    return traces, offsets, shifts, weights


@fixture
def data():
    return get_data(n_traces=100)


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_delay_sum_result_length(
    data,
    n_threads: int,
    rounds: int = ROUNDS,
    result_samples: int = 1000,
):
    traces, offsets, shifts, weights = data

    res = None
    for _ in range(rounds):
        res, offset = delay_sum.delay_sum(
            traces,
            offsets,
            shifts,
            weights,
            shift_range=(0, result_samples),
            stack=res,
            n_threads=n_threads,
        )
    qseek_max_idx, qseek_max = array_tools.argmax_masked(
        res,
        n_threads=n_threads,
    )

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
            lengthout=result_samples,
            dtype=np.float32,
        )
    pyrocko_max_idx, pyrocko_max = array_tools.argmax_masked(
        res,
        n_threads=n_threads,
    )

    assert pyrocko_max.size == result_samples

    np.testing.assert_allclose(qseek_max, pyrocko_max, rtol=1e-5)
    np.testing.assert_equal(qseek_max_idx, pyrocko_max_idx)


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_delay_sum_reduce_snapshot(data, n_threads: int):
    traces, offsets, shifts, weights = data
    max_res, max_node_idx, offset = delay_sum.delay_sum_reduce(
        traces,
        offsets,
        shifts,
        weights,
        n_threads=n_threads,
    )

    res_pyrocko, offset_pyrocko = pyrocko_parstack.parstack(
        traces,
        offsets,
        shifts,
        weights,
        result=None,
        method=0,
        nparallel=n_threads,
        dtype=np.float32,
    )
    pyrocko_max_node_idx, pyrocko_max_reduce = array_tools.argmax_masked(
        res_pyrocko,
        n_threads=n_threads,
    )

    assert offset == offset_pyrocko
    np.testing.assert_allclose(max_res, pyrocko_max_reduce, rtol=1e-5)
    np.testing.assert_equal(max_node_idx, pyrocko_max_node_idx)

    for idx in random.choices(range(res_pyrocko.shape[1]), k=1000):
        snapshot = delay_sum.delay_sum_snapshot(
            traces,
            offsets,
            shifts,
            weights,
            index=idx,
        )
        pyrocko_max_reduce_snapshot = res_pyrocko[:, idx]
        np.testing.assert_allclose(snapshot, pyrocko_max_reduce_snapshot, rtol=1e-5)


@pytest.mark.benchmark(group="delay_sum")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
@pytest.mark.parametrize("implementation", get_args(Implementation))
def test_delay_sum(
    benchmark,
    data,
    n_threads: int,
    implementation: Implementation,
    rounds: int = ROUNDS,
):
    traces, offsets, shifts, weights = data

    def stack_qseek() -> tuple[np.ndarray, int]:
        res = None
        for _ in range(rounds):
            res, offset = delay_sum.delay_sum(
                traces,
                offsets,
                shifts,
                weights,
                stack=res,
                n_threads=n_threads,
            )
        array_tools.argmax_masked(
            res,
            n_threads=n_threads,
        )
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
        array_tools.argmax_masked(
            res,
            n_threads=n_threads,
        )
        return res, offset

    def benchmark_if(func: Callable, func_implementation: Implementation):
        return benchmark(func) if func_implementation == implementation else func()

    r_qseek, off_qseek = benchmark_if(stack_qseek, "qseek")
    r_pyrocko, off_pyrocko = benchmark_if(stack_pyrocko, "pyrocko")

    np.testing.assert_allclose(r_pyrocko, r_qseek, rtol=1e-5)

    assert off_pyrocko == off_qseek


@pytest.mark.benchmark(group="delay_sum_reduce")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
@pytest.mark.parametrize("implementation", get_args(Implementation))
def test_delay_sum_reduce(
    benchmark, data, n_threads: int, implementation: Implementation
):
    def stack_reduce_qseek():
        traces, offsets, shifts, weights = data
        max_value, max_idx, offset = delay_sum.delay_sum_reduce(
            traces,
            offsets,
            shifts,
            weights,
            n_threads=n_threads,
        )
        return max_idx, max_value, offset

    def stack_reduce_pyrocko():
        traces, offsets, shifts, weights = data
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

        max_idx, max_value = array_tools.argmax_masked(
            res,
            n_threads=n_threads,
        )
        return max_idx, max_value, offset

    def benchmark_if(func: Callable, func_implementation: Implementation):
        return benchmark(func) if func_implementation == implementation else func()

    qseek_max_idx, qseek_max, qseek_off = benchmark_if(stack_reduce_qseek, "qseek")
    pyr_max_idx, pyr_max, pyr_off = benchmark_if(stack_reduce_pyrocko, "pyrocko")

    np.testing.assert_allclose(pyr_max, qseek_max, rtol=1e-5)
    np.testing.assert_equal(pyr_max_idx, qseek_max_idx)

    assert pyr_off == qseek_off


@pytest.mark.benchmark(group="delay_sum_reduce_mask")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
@pytest.mark.parametrize("implementation", ["pyrocko", "qseek"])
def test_delay_sum_reduce_mask(
    benchmark,
    data,
    n_threads: int,
    implementation: Literal["pyrocko", "qseek"],
):
    traces, offsets, shifts, weights = data
    n_nodes = shifts.shape[0]

    # Mask random 10 nodes
    mask = np.zeros(n_nodes, dtype=bool)
    masked_indices = random.sample(range(n_nodes), 10)
    mask[masked_indices] = True

    def stack_reduce_qseek():
        return delay_sum.delay_sum_reduce(
            traces,
            offsets,
            shifts,
            weights,
            node_mask=mask,
            node_stack_max=None,
            node_stack_max_idx=None,
            n_threads=n_threads,
        )

    def stack_reduce_pyrocko():
        weights_masked = np.where(mask[:, np.newaxis], 0.0, weights)
        res, offset = pyrocko_parstack.parstack(
            traces,
            offsets,
            shifts,
            weights_masked,
            result=None,
            method=0,
            nparallel=n_threads,
            dtype=np.float32,
        )

        max_idx, max_value = array_tools.argmax_masked(
            res,
            n_threads=n_threads,
        )
        return max_idx, max_value, offset

    def benchmark_if(func: Callable, func_implementation: Implementation):
        return benchmark(func) if func_implementation == implementation else func()

    max_value, max_idx, offset = benchmark_if(stack_reduce_qseek, "qseek")
    pyr_max_idx, pyr_max, pyr_off = benchmark_if(stack_reduce_pyrocko, "pyrocko")

    np.testing.assert_allclose(pyr_max, max_value, rtol=1e-5)
    np.testing.assert_equal(pyr_max_idx, max_idx)

    assert pyr_off == offset


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_delay_sum_reduce_mask_iterative(data, n_threads: int):
    traces, offsets, shifts, weights = data
    n_nodes = shifts.shape[0]

    rng = np.random.default_rng(123)

    # Mask random 10 nodes
    mask = np.zeros(n_nodes, dtype=bool)
    masked_indices = rng.choice(range(n_nodes), size=n_nodes // 2)
    mask[masked_indices] = True

    # Reference implementation
    pyr_res, pyr_offset = delay_sum.delay_sum(
        traces,
        offsets,
        shifts,
        weights,
        stack=None,
        n_threads=n_threads,
    )
    pyr_max_idx, pyr_max = array_tools.argmax_masked(pyr_res, n_threads=n_threads)

    stack_max, stack_idx, offset = delay_sum.delay_sum_reduce(
        traces,
        offsets,
        shifts,
        weights,
        node_mask=mask,
        node_stack_max=None,
        node_stack_max_idx=None,
        n_threads=n_threads,
    )

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pyr_max, stack_max, rtol=1e-5)

    with pytest.raises(AssertionError):
        np.testing.assert_equal(pyr_max_idx, stack_idx)

    # With reversed mask and results from above
    stack_max, stack_idx, offset = delay_sum.delay_sum_reduce(
        traces,
        offsets,
        shifts,
        weights,
        node_mask=~mask,
        node_stack_max=stack_max,
        node_stack_max_idx=stack_idx,
        n_threads=n_threads,
    )

    np.testing.assert_allclose(pyr_max, stack_max, rtol=1e-5)
    np.testing.assert_equal(pyr_max_idx, stack_idx)

    assert pyr_offset == offset
