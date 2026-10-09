from collections.abc import Sequence

import numpy as np

from qseek.reduce import NodeStack, TraceInput

def delay_sum(
    traces: Sequence[TraceInput],
    nodes: Sequence[NodeStack],
    stack: np.ndarray | None = None,
    shift_range: tuple[int, int] | None = None,
    n_threads: int = 1,
) -> tuple[np.ndarray, int]:
    """Beamforming of seismic traces by delay and sum.

    Args:
        traces (Sequence[TraceInput]): Traces, each with `float32` samples and
            its offset in samples.
        nodes (Sequence[NodeStack]): Nodes to stack, each with `int32` shifts
            and `float32` weights of length `n_traces`.
        stack (np.ndarray | None, optional): Result array of shape
            `(n_nodes, n_samples)`, dtype `float32`, which the stacks are added
            to. If `None`, a new array is created. Defaults to `None`.
        shift_range (tuple[int, int] | None, optional): Range of the result in
            samples. If `None`, the full range of the shifted traces is used.
            Defaults to `None`.
        n_threads (int, optional): Number of threads, 0 for all cores.
            Defaults to 1.

    Returns:
        tuple[np.ndarray, int]: Stack of each node, shape `(n_nodes, n_samples)`,
            and the minimum shift in samples.
    """

def delay_sum_reduce(
    traces: Sequence[TraceInput],
    nodes: Sequence[NodeStack],
    shift_range: tuple[int, int] | None = None,
    node_stack_max: np.ndarray | None = None,
    node_stack_max_idx: np.ndarray | None = None,
    n_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Beamforming by delay and sum, reduced to the maximum over the nodes.

    Args:
        traces (Sequence[TraceInput]): Traces, each with `float32` samples and
            its offset in samples.
        nodes (Sequence[NodeStack]): Nodes to stack, each with `int32` shifts
            and `float32` weights of length `n_traces`. The argmax records
            `NodeStack.index`.
        shift_range (tuple[int, int] | None, optional): Range of the result in
            samples. If `None`, the full range of the shifted traces is used.
            Defaults to `None`.
        node_stack_max (np.ndarray | None, optional): Running maximum of shape
            `(n_samples,)`, dtype `float32`, updated in place. If `None`, a new
            array is created. Defaults to `None`.
        node_stack_max_idx (np.ndarray | None, optional): Node index of the
            running maximum, shape `(n_samples,)`, dtype `int32`, updated in
            place. Given together with `node_stack_max`. Defaults to `None`.
        n_threads (int, optional): Number of threads, 0 for all cores.
            Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Maximum and its node index per
            sample, and the minimum shift in samples.
    """

def delay_sum_snapshot(
    traces: Sequence[TraceInput],
    nodes: Sequence[NodeStack],
    index: int,
    shift_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Delay and sum of each node at a single sample.

    Args:
        traces (Sequence[TraceInput]): Traces, each with `float32` samples and
            its offset in samples.
        nodes (Sequence[NodeStack]): Nodes to stack, each with `int32` shifts
            and `float32` weights of length `n_traces`.
        index (int): Sample index in the result.
        shift_range (tuple[int, int] | None, optional): Range of the result in
            samples. If `None`, the full range of the shifted traces is used.
            Defaults to `None`.

    Returns:
        np.ndarray: Delay and sum of each node, aligned with `nodes`.
    """
