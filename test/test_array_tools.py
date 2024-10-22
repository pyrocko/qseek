import numpy as np

from qseek.ext import array_tools


def test_fill_zero_bytes() -> None:
    nnodes = 1000
    nsamples = 3000
    data = np.ones(shape=(nnodes, nsamples), dtype=np.float32)
    zeros = np.zeros(shape=(nnodes, nsamples), dtype=np.float32)

    array_tools.fill_zero_bytes(data)
    np.testing.assert_array_equal(data, zeros)


def test_argmax():
    nnodes = 1000
    nsamples = 3000
    data = np.random.uniform(size=(nnodes, nsamples)).astype(np.float32)

    idx, value = array_tools.argmax_masked(data)

    np.testing.assert_array_equal(np.argmax(data, axis=0), idx)
    np.testing.assert_array_equal(np.max(data, axis=0), value)

    mask = np.random.choice([0, 1], size=nnodes, p=[0.5, 0.5]).astype(bool)
    idx, value = array_tools.argmax_masked(data, mask=mask)

    np.testing.assert_array_equal(np.max(data[mask], axis=0), value)
