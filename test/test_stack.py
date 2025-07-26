import numpy as np
import pytest
from pyrocko import parstack as pyrocko_parstack
from pytest import fixture

from qseek.ext import array_tools, stack

try:
    from qseek.ext_mojo import stack as stack_mojo
except ImportError:
    stack_mojo = None

N_THREADS_TEST = [1, 2, 4, 8]


def get_data(n_nodes: int = 12 * 12 * 10, n_samples: int = 30_000, n_traces: int = 100):
    # n_nodes = 20
    # n_samples = 500
    # n_traces = 1

    traces = []
    for _ in range(n_traces):
        traces.append(np.random.uniform(0, 32000, (n_samples)).astype(np.float32))

    offsets = np.random.randint(-10, 10, n_traces, dtype=np.int32)
    shifts = np.random.randint(-100, 100, size=(n_nodes, n_traces)).astype(np.int32)
    weights = np.ones((n_nodes, n_traces), dtype=np.float32)

    return traces, offsets, shifts, weights


@fixture
def data():
    return get_data(n_traces=100)


@fixture
def data_double():
    return get_data(n_traces=200)


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack(data, n_threads: int):
    traces, offsets, shifts, weights = data
    res, offset = stack.stack_traces(
        traces,
        offsets,
        shifts,
        weights,
        result=None,
        n_threads=n_threads,
    )
    res_pyrocko, offset_pyrocko = pyrocko_parstack.parstack(
        traces,
        offsets,
        shifts,
        weights,
        dtype=np.float32,
        method=0,
        nparallel=n_threads,
    )

    np.testing.assert_allclose(res, res_pyrocko, rtol=1e-5)
    assert offset == offset_pyrocko


@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack_and_reduce(data, n_threads: int):
    traces, offsets, shifts, weights = data
    max_res, max_node_idx, offset = stack.stack_and_reduce(
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


@pytest.mark.benchmark(group="parstack")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack_qseek(benchmark, data, n_threads: int):
    traces, offsets, shifts, weights = data

    benchmark(
        stack.stack_traces,
        traces,
        offsets,
        shifts,
        weights,
        result=None,
        result_samples=20_000,
        n_threads=n_threads,
    )


@pytest.mark.benchmark(group="parstack")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack_pyrocko(benchmark, data, n_threads: int):
    traces, offsets, shifts, weights = data

    benchmark(
        pyrocko_parstack.parstack,
        traces,
        offsets,
        shifts,
        weights,
        method=0,
        nparallel=n_threads,
    )


@pytest.mark.benchmark(group="parstack")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def _test_stack_qseek_mojo(benchmark, data, n_threads: int):
    traces, offsets, shifts, weights = data

    benchmark(
        stack_mojo.stack_traces,
        traces,
        offsets,
        shifts,
        weights,
        n_threads=n_threads,
    )


@pytest.mark.benchmark(group="parstack_reduce")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack_reduce_qseek(benchmark, data_double, n_threads: int):
    traces, offsets, shifts, weights = data_double
    benchmark(
        stack.stack_and_reduce,
        traces,
        offsets,
        shifts,
        weights,
        n_threads=n_threads,
    )


@pytest.mark.benchmark(group="parstack_reduce")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def _test_stack_reduce_qseek_mojo(benchmark, data_double, n_threads: int):
    traces, offsets, shifts, weights = data_double
    benchmark(
        stack_mojo.stack_and_reduce,
        traces,
        offsets,
        shifts,
        weights,
        n_threads=n_threads,
    )


@pytest.mark.benchmark(group="parstack_reduce")
@pytest.mark.parametrize("n_threads", N_THREADS_TEST)
def test_stack_reduce_pyrocko(benchmark, data, n_threads: int):
    traces, offsets, shifts, weights = data

    @benchmark
    def reduce():
        res, _ = pyrocko_parstack.parstack(
            traces,
            offsets,
            shifts,
            weights,
            result=None,
            method=0,
            nparallel=n_threads,
            dtype=np.float32,
        )
        res, _ = pyrocko_parstack.parstack(
            traces,
            offsets,
            shifts,
            weights,
            result=res,
            method=0,
            nparallel=n_threads,
            dtype=np.float32,
        )

        array_tools.argmax_masked(
            res,
            n_threads=n_threads,
        )

    assert reduce is None
