import numpy as np

def fill_zero_bytes(array: np.ndarray, n_threads: int = 8) -> None:
    """Fill the zero bytes of the array with zeros.

    This function is 2x faster compared to the `data.fill(0.0)` method.

    Args:
        array: The array to fill with zero bytes.
        n_threads: The number of threads to use. Default is 8.
    """

def argmax_masked(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    n_threads: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the argmax of the data array.

    Args:
        data: The data array, ndim=2 with NxM shape.
        mask: The mask array, ndim=1 with N shape of np.bool type. Default is None.
        n_threads: The number of threads to use. Default is 8.

    Returns:
        The tuple of the argmax index and the value.
    """
