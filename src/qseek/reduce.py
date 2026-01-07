from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from pyrocko.trace import Trace
from scipy import signal

from qseek.ext.delay_sum import delay_sum_reduce, delay_sum_snapshot
from qseek.stats import Stats

if TYPE_CHECKING:
    from qseek.octree import Node

logger = logging.getLogger(__name__)


class DelaySumReduceStats(Stats): ...


STATS = DelaySumReduceStats()


class DelaySumReduce:
    traces: list[Trace]
    nodes: list[Node]

    _start_time: datetime
    _end_time: datetime
    _padding: timedelta
    _sampling_rate: float

    _trace_offsets: np.ndarray
    _trace_weights: np.ndarray
    _node_shifts: np.ndarray

    _trace_data: list[np.ndarray]
    _padding_samples: int
    _result_nsamples: int
    _stack_offset: int

    _stack_max: np.ndarray
    _stack_max_idx: np.ndarray

    _node_idx: dict[bytes, int]
    _node_removal_mask: np.ndarray
    _integrated_nodes: np.ndarray
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

        padded_start_time = start_time - padding
        trace_tmins = np.fromiter((tr.tmin for tr in traces), float)
        self._trace_offsets = np.round(
            (trace_tmins - padded_start_time.timestamp()) * sr
        ).astype(np.int32)

        self.traces = traces
        self.n_traces = len(traces)
        self._start_time = start_time
        self._end_time = end_time
        self._padding = padding

        self._sampling_rate = sr
        self._node_shifts = np.empty((0, self.n_traces), dtype=np.int32)
        self._trace_weights = np.empty((0, self.n_traces), dtype=np.float32)

        self._trace_data = [tr.ydata.astype(np.float32, copy=False) for tr in traces]

        self._stack_max = np.zeros(self._result_nsamples, dtype=np.float32)
        self._stack_max_idx = np.zeros(self._result_nsamples, dtype=int)

        self._node_idx = {}
        self._node_removal_mask = np.empty(0, dtype=bool)
        self._integrated_nodes = np.empty(0, dtype=bool)

        self.nodes = []

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def _invalidate_state(self) -> None:
        self._dirty = True

    def _check_state(self) -> None:
        if self._dirty:
            raise EnvironmentError(
                "Stack is dirty, please recompute by calling stack() before use."
            )

    def remove_nodes(self, nodes: Sequence[Node]) -> None:
        """Remove nodes from stacking.

        Args:
            nodes (list[Node]): Nodes to remove.

        Raises:
            ValueError: If one or more nodes are not found.
        """
        try:
            indices = np.array([self._node_idx[node.hash] for node in nodes])
        except KeyError as e:
            raise ValueError("One or more nodes to remove not found.") from e
        self._node_removal_mask[indices] = True
        self._stack_max[np.isin(self._stack_max_idx, indices)] = 0.0
        self._invalidate_state()

    def add_nodes(
        self,
        nodes: Sequence[Node],
        traveltimes: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        n_new_nodes = len(nodes)

        required_shape = (n_new_nodes, self.n_traces)
        if traveltimes.shape != required_shape:
            raise ValueError(f"Shifts shape must be {required_shape}.")
        if weights.shape != required_shape:
            raise ValueError(f"Weights shape must be {required_shape}.")

        if weights.dtype != np.float32:
            raise ValueError("Weights must be of dtype np.float32.")

        # Cleaning traveltimes
        traveltime_mask = np.isnan(traveltimes)
        traveltimes[traveltime_mask] = 0.0
        weights[traveltime_mask] = 0.0

        shifts = np.round(-traveltimes * self._sampling_rate).astype(np.int32)

        self._node_shifts = np.vstack((self._node_shifts, shifts))
        self._trace_weights = np.vstack((self._trace_weights, weights))

        self._node_removal_mask = np.concatenate(
            (self._node_removal_mask, np.zeros(n_new_nodes, dtype=bool)),
        )
        self._integrated_nodes = np.concatenate(
            (self._integrated_nodes, np.zeros(n_new_nodes, dtype=bool))
        )

        n_nodes_old = len(self.nodes)
        new_indices = {node.hash: n_nodes_old + i for i, node in enumerate(nodes)}
        self._node_idx.update(new_indices)

        self.nodes.extend(nodes)
        self._invalidate_state()

    async def stack(
        self,
        n_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the stacked traces and corresponding maxima stack and node indices.

        Args:
            n_threads (int, optional): Number of threads to use. Defaults to 0.
            nodes (list[Node] | None, optional): Nodes to include in stacking.
                If None, all nodes are included. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Unpadded stacked maximum values and
                node indices.
        """
        rq_shp = (self.n_nodes, self.n_traces)
        if self._node_shifts.shape != rq_shp or self._trace_weights.shape != rq_shp:
            raise ValueError(f"Shape of weights and shifts must be {rq_shp}.")

        (
            self._stack_max,
            self._stack_max_idx,
            self._stack_offset,
        ) = await asyncio.to_thread(
            delay_sum_reduce,
            traces=self._trace_data,
            offsets=self._trace_offsets,
            shifts=self._node_shifts,
            weights=self._trace_weights,
            node_mask=self._integrated_nodes,
            stack_range=(0, self._result_nsamples),
            node_stack_max=self._stack_max,
            node_stack_max_idx=self._stack_max_idx,
            n_threads=n_threads,
        )
        self._integrated_nodes[:] = True
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
        if trim_padding:
            begin = self._padding_samples
            end = -(self._padding_samples + 1)
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

    async def get_snapshot(self, sample: int) -> np.ndarray:
        """Get a snapshot of the delay-sum at a given sample index.

        Removed nodes are excluded from the snapshot.

        Args:
            sample (int): Sample index to get the snapshot at.

        Returns:
            np.ndarray: Snapshot of the delay-sum at the given sample index.
        """
        snapshot = delay_sum_snapshot(
            traces=self._trace_data,
            offsets=self._trace_offsets,
            shifts=self._node_shifts,
            weights=self._trace_weights,
            index=sample + self._padding_samples,
            stack_range=(0, self._result_nsamples),
            node_mask=self._node_removal_mask,
        )
        return snapshot[~self._node_removal_mask]

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
