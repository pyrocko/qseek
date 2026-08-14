from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import mojo.importer  # noqa: F401  (installs the import hook used just below)
import numpy as np
from pyrocko.trace import Trace
from scipy import signal

from qseek.ext_mojo import delay_sum as _delay_sum
from qseek.stats import Stats

if TYPE_CHECKING:
    from qseek.octree import Node

logger = logging.getLogger(__name__)


class TraceInput(NamedTuple):
    """One trace as handed to the Mojo kernels: samples plus their offset.

    `offset` is where sample 0 sits in the padded result window, so the
    kernels need no separate offsets array alongside the trace list.
    """

    data: np.ndarray
    offset: int


class NodeStack(NamedTuple):
    """One grid node's delay-and-sum inputs, as handed to the Mojo kernels.

    Named `NodeStack` -- not `Node` -- to stay distinct from
    `qseek.octree.Node`: there is one `NodeStack` per octree node, but it
    only carries what the kernels need to stack it: which slot in the output
    arrays it owns (`index`, i.e. its position in `DelaySumReduce.nodes`)
    and its per-trace shift/weight.

    Together with `TraceInput`, this is the whole interface to
    `qseek/ext_mojo/delay_sum.mojo`: two plain lists, where the C extension
    took packed 2-D arrays plus a parallel offsets array and a node mask.
    Mojo's `Grid` just reads attributes off whatever lists it is given.
    """

    index: int
    shifts: np.ndarray
    weights: np.ndarray


class DelaySumReduceStats(Stats): ...


STATS = DelaySumReduceStats()


class DelaySumReduce:
    traces: list[Trace]
    nodes: list[Node]

    _start_time: datetime
    _padding: timedelta
    _sampling_rate: float

    _trace_inputs: list[TraceInput]
    _node_stacks: list[NodeStack]

    _padding_samples: int
    _result_nsamples: int
    _stack_offset: int

    _stack_max: np.ndarray
    _stack_max_idx: np.ndarray

    _node_idx: dict[bytes, int]
    _n_stacked: int
    _dirty: bool = True

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        padding: timedelta,
        traces: list[Trace],
    ) -> None:
        if end_time <= start_time:
            raise ValueError("End time must be after start time.")
        sampling_rates = {1.0 / tr.deltat for tr in traces}
        if len(sampling_rates) != 1:
            raise ValueError("All traces must have the same sampling rate.")

        sr = sampling_rates.pop()

        self._padding_samples = round(padding.total_seconds() * sr)
        self._result_nsamples = (
            int((end_time - start_time).total_seconds() * sr)
            + 2 * self._padding_samples
        )
        self._stack_offset = 0

        self.traces = traces
        self.n_traces = len(traces)
        self._start_time = start_time
        self._padding = padding

        self._sampling_rate = sr
        self._node_stacks = []

        # `ascontiguousarray` rather than `astype(copy=False)`: the latter
        # hands back a strided array unchanged when the dtype already
        # matches, and the kernels read the raw buffer as if it were packed.
        padded_start = (start_time - padding).timestamp()
        self._trace_inputs = [
            TraceInput(
                data=np.ascontiguousarray(tr.ydata, dtype=np.float32),
                offset=round((tr.tmin - padded_start) * sr),
            )
            for tr in traces
        ]

        self._stack_max = np.zeros(self._result_nsamples, dtype=np.float32)
        self._stack_max_idx = np.zeros(self._result_nsamples, dtype=np.int32)

        self._node_idx = {}
        self._n_stacked = 0

        self.nodes = []

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def _invalidate_state(self) -> None:
        self._dirty = True

    def _check_state(self) -> None:
        if self._dirty:
            raise RuntimeError(
                "Stack is dirty, please recompute by calling stack() before use."
            )

    def remove_nodes(self, nodes: Sequence[Node]) -> None:
        """Retract nodes' contributions from the running maximum stack.

        Samples currently won by one of `nodes` are reset to 0.0, so the
        next `stack()` re-derives them from the nodes added since. The nodes
        themselves stay in `self.nodes` and keep their indices; only their
        wins are dropped.

        Note that this is an approximation: samples are reset rather than
        recomputed against the remaining nodes, so a sample whose runner-up
        is an older node is re-derived from the newly added nodes alone.
        The caller (search.py) splits a node and immediately adds its
        children, which cover the same region.

        Args:
            nodes (Sequence[Node]): Nodes to retract.

        Raises:
            ValueError: If one or more nodes are not found.
        """
        try:
            indices = np.array([self._node_idx[node.hash] for node in nodes])
        except KeyError as e:
            raise ValueError("One or more nodes to remove not found.") from e
        self._stack_max[np.isin(self._stack_max_idx, indices)] = 0.0
        self._invalidate_state()

    def add_nodes(
        self,
        nodes: list[Node],
        traveltimes: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        n_new_nodes = len(nodes)

        required_shape = (n_new_nodes, self.n_traces)
        if traveltimes.shape != required_shape:
            raise ValueError(f"Traveltimes shape must be {required_shape}.")
        if weights.shape != required_shape:
            raise ValueError(f"Weights shape must be {required_shape}.")

        if weights.dtype != np.float32:
            raise ValueError("Weights must be of dtype np.float32.")

        # A NaN traveltime means the phase never arrives at that station;
        # zeroing its weight makes the kernel skip the pair entirely. Both
        # results are fresh arrays -- `traveltimes` and `weights` belong to
        # the caller and must not be written through.
        no_arrival = np.isnan(traveltimes)
        shifts = np.round(
            -np.where(no_arrival, 0.0, traveltimes) * self._sampling_rate
        ).astype(np.int32)
        weights = np.where(no_arrival, np.float32(0.0), weights)

        # Nodes are only ever appended, so a node's index is its position in
        # both `self.nodes` and `self._node_stacks`. `NodeStack.index` still
        # has to be stored, because `stack()` hands the kernel a *subset* of
        # `_node_stacks` and each item must name its own global slot.
        #
        # The rows are views into the two arrays built above, which this
        # object now solely owns: one allocation per call, rows contiguous
        # and adjacent, rather than a small copy per node.
        n_nodes_old = len(self.nodes)
        self._node_stacks.extend(
            NodeStack(index=n_nodes_old + i, shifts=shifts[i], weights=weights[i])
            for i in range(n_new_nodes)
        )

        self._node_idx.update(
            {node.hash: n_nodes_old + i for i, node in enumerate(nodes)}
        )
        self.nodes.extend(nodes)
        self._invalidate_state()

    async def stack(
        self,
        n_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold any newly added nodes into the running maximum stack.

        Accumulation is incremental: nodes already stacked by a previous
        call stay folded into `_stack_max`/`_stack_max_idx`, so only the
        nodes appended since then are passed to the kernel.

        Args:
            n_threads (int, optional): Number of threads to use. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]: Padded stacked maximum values and
                node indices. Use `get_stack()` to trim the padding.
        """
        new_node_stacks = self._node_stacks[self._n_stacked :]
        if new_node_stacks:
            (
                self._stack_max,
                self._stack_max_idx,
                self._stack_offset,
            ) = await asyncio.to_thread(
                _delay_sum.delay_sum_reduce,
                self._trace_inputs,
                new_node_stacks,
                shift_range=(0, self._result_nsamples),
                node_stack_max=self._stack_max,
                node_stack_max_idx=self._stack_max_idx,
                n_threads=n_threads,
            )
            self._n_stacked = len(self._node_stacks)

        self._dirty = False

        return self._stack_max, self._stack_max_idx

    @property
    def start_time(self) -> datetime:
        return self._start_time + timedelta(
            seconds=self._stack_offset / self._sampling_rate
        )

    def get_stack(self, trim_padding: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Get stacked maximum values and corresponding node indices.

        Returns:
            tuple[np.ndarray, np.ndarray]: Stacked maximum values and node indices.
        """
        self._check_state()
        if trim_padding and self._padding_samples:
            begin = self._padding_samples
            end = self._stack_max.size - self._padding_samples
            return self._stack_max[begin:end], self._stack_max_idx[begin:end]

        return self._stack_max, self._stack_max_idx

    async def get_trace(self, trim_padding: bool = True) -> Trace:
        """Get aggregated maximum semblance as a Pyrocko trace.

        Returns:
            Trace: Holding the semblance.
        """
        self._check_state()
        data, _ = self.get_stack(trim_padding)
        start_time = (
            self.start_time if trim_padding else self.start_time - self._padding
        )

        return Trace(
            network="",
            station="semblance",
            tmin=start_time.timestamp(),
            deltat=1.0 / self._sampling_rate,
            ydata=data,
        )

    async def get_snapshot(self, sample: int, leaf_only: bool = True) -> np.ndarray:
        """Get a snapshot of the delay-sum at a given sample index.

        Args:
            sample (int): Sample index to get the snapshot at.
            leaf_only (bool, optional): If True, only leaf nodes (nodes
                without children) are included in the snapshot. Defaults to
                True.

        Returns:
            np.ndarray: Snapshot of the delay-sum at the given sample index,
                aligned with `self.nodes` (or its leaf-only subset).
        """
        node_stacks = self._node_stacks
        if leaf_only:
            node_stacks = [
                node_stack
                for node_stack, node in zip(self._node_stacks, self.nodes, strict=True)
                if not node.children
            ]

        return _delay_sum.delay_sum_snapshot(
            self._trace_inputs,
            node_stacks,
            index=sample + self._padding_samples,
            shift_range=(0, self._result_nsamples),
        )

    async def find_peaks(
        self,
        height: float,
        prominence: float,
        distance: float,
        trim_padding: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find peaks in maximum semblance.

        For details see scipy.signal.find_peaks.

        Args:
            height (float): Minimum height of the peak.
            prominence (float): Prominence of the peak.
            distance (float): Minium distance of a peak to other peaks.
            trim_padding (bool, optional): Trim padded data in post-processing.
                Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices of peaks and peak values.
        """
        self._check_state()
        stack_max, _ = self.get_stack(trim_padding=False)

        detection_idx, _ = await asyncio.to_thread(
            signal.find_peaks,
            stack_max,
            height=height,
            prominence=prominence,
            distance=distance,
        )
        if trim_padding:
            stack_max_trimmed, _ = self.get_stack(trim_padding=True)

            detection_idx -= self._padding_samples
            detection_idx = detection_idx[detection_idx >= 0]
            detection_idx = detection_idx[detection_idx < stack_max_trimmed.size]
            semblance = stack_max_trimmed[detection_idx]
        else:
            semblance = stack_max[detection_idx]

        return detection_idx, semblance

    def get_time_from_sample(self, sample: int) -> datetime:
        """Get the time from a sample index.

        Args:
            sample (int): The sample index.

        Returns:
            datetime: The time of the sample.
        """
        return self.start_time + timedelta(seconds=sample / self._sampling_rate)
