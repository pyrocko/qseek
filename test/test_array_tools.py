import numpy as np

from qseek.ext import array_tools


def test_argmax():
    rng = np.random.default_rng(0)
    nnodes = 1000
    nsamples = 3000
    data = rng.uniform(size=(nnodes, nsamples)).astype(np.float32)

    idx, value = array_tools.argmax_masked(data)

    np.testing.assert_array_equal(np.argmax(data, axis=0), idx)
    np.testing.assert_array_equal(np.max(data, axis=0), value)

    mask = rng.choice([0, 1], size=nnodes, p=[0.5, 0.5]).astype(bool)
    idx, value = array_tools.argmax_masked(data, mask=mask)

    np.testing.assert_array_equal(np.max(data[mask], axis=0), value)
