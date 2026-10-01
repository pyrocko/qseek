import numpy as np

from qseek.delay_sum import NodeStack

def delay_sum(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    nodes: list[NodeStack],
    stack: np.ndarray | None = None,
    shift_range: tuple[int, int] | None = None,
    n_threads: int = 1,
) -> tuple[np.ndarray, int]:
    """Beamforming of seismic trace by delay and sum method.

    This implements a delay-and-sum beamforming algorithm optimized with OpenMP and SIMD
    to backproject seismic energy into a irregular subsurface grid of N-nodes.

    Args:
        traces (list[np.ndarray]): List of seismic traces as numpy arrays
            of type `np.float32`.
        offsets (np.ndarray): Static offsets of seismic traces in samples for each node.
            Shape is `(n_nodes, n_traces)`, dtype is `int`.
        nodes (list[NodeStack]): List of NodeStack namedtuples containing shifts,
            weights, mask and trace group for each node.
        stack (np.ndarray | None, optional): The resulting stack array of size
        `n_nodes, n_samples`, dtype is `np.float32`. If `None` a new array will
            be created of size `(n_nodes, n_samples)`. Defaults to `None`.
        shift_range (tuple[int, int] | None, optional): Stack range of shifts in
        samples. If `None`, the full range of shifts will be used.
            Defaults to `None`.
        n_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        tuple[np.ndarray, int]: Tuple of stacked traces as numpy array of shape
            `(n_nodes, n_samples)`, dtype `np.float32` and minimum offset in samples.
    """

def delay_sum_reduce(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    nodes: list[NodeStack],
    shift_range: tuple[int, int] | None = None,
    node_stack_max: np.ndarray | None = None,
    node_stack_max_idx: np.ndarray | None = None,
    n_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Beamforming of seismic trace by delay and sum method with max semblancereduction.

    This implements a delay-and-sum beamforming algorithm optimized with OpenMP and SIMD
    to backproject seismic energy into a irregular subsurface grid of N-nodes.
    In addition, the semblance is reduced to the maximum to obtain the maximum semblance
    at each time step.

    Args:
        traces (list[np.ndarray]): List of seismic traces as numpy arrays
            of type `np.float32`.
        offsets (np.ndarray): Static offsets of seismic traces in samples for each node.
            Shape is `(n_nodes, n_traces)`, dtype is `int`.
        nodes (list[NodeStack]): List of NodeStack namedtuples containing shifts,
            weights, mask and trace group for each node.
        shift_range (tuple[int, int] | None, optional): Stack range of shifts in
            samples. If `None`, the full range of shifts will be used.
            Defaults to `None`.
        node_stack_max (np.ndarray | None, optional): Array to store the maximum
            stack value for each node of size `n_nodes`, dtype is `np.float32`.
        node_stack_max_idx (np.ndarray | None, optional): Array to store the index
            of the maximum stack value for each node of size `n_nodes`, dtype is `int`.
            Defaults to `None`.
        n_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Tuple of maximum reduced stacked semblance
            as numpy array of size `n_samples` as dtype `np.float32`, array of node
            indices of maximum semblance of size `n_samples`, dtype `int`,
            and minimum offset in samples.
    """

def delay_sum_snapshot(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    nodes: list[NodeStack],
    index: int,
    shift_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Snapshot of delay and sum in time at a specified sample index.

    Make a snapshot of the delay-and-sum beamforming at a specified sample index
    for all nodes.

    Args:
        traces (list[np.ndarray]): List of seismic traces as numpy arrays
            of type `np.float32`.
        offsets (np.ndarray): Static offsets of seismic traces in samples for each node.
            Shape is `(n_nodes, n_traces)`, dtype is `int`.
        nodes (list[NodeStack]): List of NodeStack namedtuples containing shifts,
            weights, mask and trace group for each node.
        index (int): Sample index to make the snapshot at.
        shift_range (tuple[int, int] | None, optional): Stack range of shifts in
            samples. If `None`, the full range of shifts will be used.
            Defaults to `None`.

    Returns:
        np.ndarray: Snapshot of the delay-sum at the given sample index of size
            `n_nodes`, dtype `np.float32`.
    """
