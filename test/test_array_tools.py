import numpy as np

from qseek.ext import array_tools


def test_fill_cache() -> None:
    nnodes = 1000
    nsamples = 3000
    data = np.ones(shape=(nnodes, nsamples), dtype=np.float32)
    data_cache = np.random.uniform(size=(nnodes, nsamples)).astype(np.float32)
    mask = np.random.choice([0, 1], size=nnodes, p=[0.5, 0.5]).astype(np.bool_)

    cache = [d for d, m in zip(data_cache, mask, strict=False) if m]

    def numpy_fill(data, cache, mask):
        data = data.copy()
        data[mask, :] = cache
        return data

    def mview(data, cache, mask):
        data = data.copy()
        cache = cache.copy()
        for idx, copy in enumerate(mask):
            if copy:
                memoryview(data[idx])[:] = memoryview(cache.pop(0))
        return data

    data_numpy = numpy_fill(data, cache, mask)
    data_mview = mview(data, cache, mask)
    array_tools.apply_cache(data, cache, mask, nthreads=1)

    np.testing.assert_array_equal(data_numpy, data_mview)
    np.testing.assert_array_equal(data_mview, data)


def test_fill_zero_bytes() -> None:
    nnodes = 1000
    nsamples = 3000
    data = np.ones(shape=(nnodes, nsamples), dtype=np.float32)
    zeros = np.zeros(shape=(nnodes, nsamples), dtype=np.float32)

    array_tools.fill_zero_bytes(data)
    np.testing.assert_array_equal(data, zeros)


def test_fill_zero_bytes_mask():
    nnodes = 1000
    nsamples = 3000
    data = np.ones(shape=(nnodes, nsamples), dtype=np.float32)
    mask = np.random.choice([0, 1], size=nnodes, p=[0.5, 0.5]).astype(np.bool_)

    array_tools.fill_zero_bytes_mask(data, mask)
    for idx, m in enumerate(mask):
        assert np.any(data[idx]) != m
