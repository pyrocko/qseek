from __future__ import annotations

import numpy as np

from qseek.ext_mojo import stack_ext as stack


def stack_traces(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    result: list | None = None,
    n_threads: int = 1,
) -> tuple[list, int]:
    return stack.stack_traces(traces, offsets, shifts, weights, result, n_threads)


def stack_and_reduce(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    n_threads: int = 1,
) -> tuple[list, list, int]:
    return stack.stack_and_reduce(traces, offsets, shifts, weights, n_threads)


def stack_snapshot(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    index: int,
) -> np.ndarray:
    return stack.stack_snapshot(traces, offsets, shifts, weights, index)
