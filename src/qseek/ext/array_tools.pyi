import numpy as np

def fill_zero_bytes(array: np.ndarray, n_threads: int = 8) -> None:
    """Fill the zero bytes of the array with zeros.

    This function is 2x faster compared to the `data.fill(0.0)` method.

    Args:
        array: The array to fill with zero bytes.
        n_threads: The number of threads to use. Default is 8.
    """

def fill_zero_bytes_mask(
    array: np.ndarray,
    mask: np.ndarray,
    n_threads: int = 8,
) -> None:
    """Fill the zero bytes of the array with zeros using numba.

    Args:
        array: The array to fill with zero bytes ndim=2, NxM.
        mask: The mask array, ndim=1 with N shape of np.bool type.
        n_threads: The number of threads to use. Default is 8.
    """

def apply_cache(
    data: np.ndarray,
    cache: list[np.ndarray],
    mask: np.ndarray,
    nthreads: int = 1,
) -> None:
    """Apply the cache to the data array.

    Main purpose of this function to do it in one go and release the GIL.

    Args:
        data: The data array, ndim=2 with NxM shape.
        cache: List of arrays with ndim=1 and M shape.
        mask: The mask array, ndim=1 with N shape of np.bool type.
        nthreads: The number of threads to use. Default is 1.
    """
