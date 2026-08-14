"""Tests for src/qseek/reduce.py's DelaySumReduce, which is Mojo-only.

`DelaySumReduce` no longer has a C fallback (see qseek.reduce): it always
calls `qseek/ext_mojo/delay_sum.mojo`, passing traces and nodes as
`TraceInput`/`NodeStack` lists rather
than the flat arrays test_semblance.py's reference computations use for
the C extension directly. This file:

- covers `get_snapshot(leaf_only=...)`, which isn't exercised anywhere else,
  by cross-checking it against a C `delay_sum_snapshot` call built from the
  same per-node data reduce.py already computed internally.
- benchmarks `DelaySumReduce.stack()` (the production entry point) against
  an equivalent direct call into the C extension, to confirm the node-list
  integration doesn't cost anything at the reduce.py level.

Requires the `modular` (Mojo 1.0) toolchain, a required dependency; skipped
if it isn't importable (e.g. an unsupported platform).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pytest
from pyrocko.trace import Trace

pytest.importorskip(
    "mojo.importer", reason="Mojo toolchain (dependency 'modular') not installed"
)

from qseek.ext import delay_sum as qseek_delay_sum
from qseek.models.station import StationInventory, StationList
from qseek.octree import Octree
from qseek.reduce import DelaySumReduce


def _make_traces(
    rng: np.random.RandomState,
    n_stations: int,
    n_samples: int,
    starttime: datetime,
    sampling_rate: float,
) -> list[Trace]:
    return [
        Trace(
            network="",
            station=f"S{i:03d}",
            ydata=rng.uniform(0, 1, n_samples).astype(np.float32),
            tmin=starttime.timestamp(),
            deltat=1.0 / sampling_rate,
        )
        for i in range(n_stations)
    ]


def _c_extension_args(stack: DelaySumReduce, node_stacks=None):
    """Rebuild the C extension's flat argument convention, for references.

    reduce.py holds `TraceInput`/`NodeStack` lists; the C extension wants
    trace arrays plus a parallel offsets array, and packed 2-D shifts and
    weights. `node_stacks` defaults to every node in `stack`.
    """
    if node_stacks is None:
        node_stacks = stack._node_stacks
    traces = [trace_input.data for trace_input in stack._trace_inputs]
    offsets = np.array(
        [trace_input.offset for trace_input in stack._trace_inputs], dtype=np.int32
    )
    shifts = np.stack([node_stack.shifts for node_stack in node_stacks])
    weights = np.stack([node_stack.weights for node_stack in node_stacks])
    return traces, offsets, shifts, weights


@pytest.mark.asyncio
async def test_reduce_get_snapshot_leaf_only(
    octree: Octree, stations: StationInventory
) -> None:
    """get_snapshot(leaf_only=True) must match a per-leaf C reference.

    Exercised after a split/remove/add cycle so leaves and non-leaves are
    both present.
    """
    rng = np.random.RandomState(7)
    station_list = StationList.from_inventory(stations)
    n_stations = station_list.n_stations
    n_nodes = octree.n_nodes

    starttime = datetime.fromisoformat("2020-01-01T00:00:00+00:00")
    endtime = starttime + timedelta(seconds=10)
    sampling_rate = 100.0
    n_samples = int((endtime - starttime).total_seconds() * sampling_rate)

    traces = _make_traces(rng, n_stations, n_samples, starttime, sampling_rate)

    shape = (n_nodes, n_stations)
    traveltimes = rng.uniform(0.0, 5.0, size=shape).astype(np.float32)
    weights = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

    stack = DelaySumReduce(
        traces=traces,
        start_time=starttime,
        end_time=endtime,
        padding=timedelta(seconds=0),
    )
    stack.add_nodes(nodes=octree.nodes, traveltimes=traveltimes, weights=weights)
    await stack.stack()

    split_node = next((n for n in octree.nodes if n.can_split()), None)
    if split_node is None:
        pytest.skip("octree not deep enough to split")

    child_nodes = split_node.split()
    child_shape = (len(child_nodes), n_stations)
    child_traveltimes = rng.uniform(0.0, 5.0, size=child_shape).astype(np.float32)
    child_weights = rng.uniform(0.0, 1.0, size=child_shape).astype(np.float32)

    stack.remove_nodes([split_node])
    stack.add_nodes(
        nodes=list(child_nodes), traveltimes=child_traveltimes, weights=child_weights
    )
    await stack.stack()

    leaf_stacks = [
        node_stack
        for node_stack, node in zip(stack._node_stacks, stack.nodes, strict=True)
        if not node.children
    ]
    assert 0 < len(leaf_stacks) < len(stack.nodes)  # both leaves and non-leaves exist
    traces_c, offsets_c, shifts, weights_flat = _c_extension_args(stack, leaf_stacks)

    sample = 100
    reference = qseek_delay_sum.delay_sum_snapshot(
        traces_c,
        offsets_c,
        shifts,
        weights_flat,
        index=sample + stack._padding_samples,
        shift_range=(0, stack._result_nsamples),
    )
    snapshot = await stack.get_snapshot(sample, leaf_only=True)

    assert snapshot.shape == (len(leaf_stacks),)
    np.testing.assert_allclose(snapshot, reference, rtol=1e-5)


# ===----------------------------------------------------------------=== #
# Benchmark: DelaySumReduce.stack() (mojo, production) vs the C extension
# ===----------------------------------------------------------------=== #

Implementation = Literal["c", "mojo"]
N_THREADS_BENCH = [1, 4]


@dataclass(slots=True)
class FakeNode:
    """Minimal stand-in for qseek.octree.Node: DelaySumReduce only needs `.hash`."""

    hash: bytes


@pytest.fixture
def benchmark_reduce_data():
    rng = np.random.RandomState(7)
    n_nodes, n_samples, n_traces = 100, 30_000, 100
    sampling_rate = 100.0

    starttime = datetime.fromisoformat("2020-01-01T00:00:00+00:00")
    endtime = starttime + timedelta(seconds=n_samples / sampling_rate)
    traces = _make_traces(rng, n_traces, n_samples, starttime, sampling_rate)
    nodes = [FakeNode(hash=i.to_bytes(8, "little")) for i in range(n_nodes)]

    shape = (n_nodes, n_traces)
    traveltimes = rng.uniform(0.0, 5.0, size=shape).astype(np.float32)
    weights = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

    return traces, nodes, traveltimes, weights, starttime, endtime


@pytest.mark.benchmark(group="reduce_stack")
@pytest.mark.parametrize("n_threads", N_THREADS_BENCH)
@pytest.mark.parametrize("implementation", ["c", "mojo"])
def test_benchmark_reduce_stack(
    benchmark, benchmark_reduce_data, implementation: Implementation, n_threads: int
):
    traces, nodes, traveltimes, weights, starttime, endtime = benchmark_reduce_data

    def build() -> DelaySumReduce:
        stack = DelaySumReduce(
            traces=traces,
            start_time=starttime,
            end_time=endtime,
            padding=timedelta(seconds=0),
        )
        stack.add_nodes(
            nodes=nodes, traveltimes=traveltimes.copy(), weights=weights.copy()
        )
        return stack

    def run_mojo() -> tuple[np.ndarray, np.ndarray]:
        stack = build()
        return asyncio.run(stack.stack(n_threads=n_threads))

    def run_c() -> tuple[np.ndarray, np.ndarray]:
        stack = build()
        traces_c, offsets_c, shifts, weights_flat = _c_extension_args(stack)
        max_value, max_idx, _ = qseek_delay_sum.delay_sum_reduce(
            traces_c,
            offsets_c,
            shifts,
            weights_flat,
            shift_range=(0, stack._result_nsamples),
            n_threads=n_threads,
        )
        return max_value, max_idx

    max_value, max_idx = benchmark(run_mojo if implementation == "mojo" else run_c)
    assert max_value.shape == max_idx.shape
