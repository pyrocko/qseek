import numpy as np

def stack_traces(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    result: np.ndarray | None = None,
    result_samples: int = 0,
    n_threads: int = 1,
) -> tuple[np.ndarray, int]: ...
def stack_and_reduce(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    n_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]: ...
def stack_snapshot(
    traces: list[np.ndarray],
    offsets: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    index: int,
) -> np.ndarray: ...
