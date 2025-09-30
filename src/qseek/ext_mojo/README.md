# Mojo Extensions

Experimental heavy-lifting functions in Mojo bringing a speedup of > 2x for stacking `parstack` and `argmax_masked`.

```
Timeit Mojo:
Min time: 0.11779344200003834
Timeit Pyrocko:
Min time: 0.29711000509996666
Speedup:  2.522296658076007
```

```py
import max._mojo.mojo_importer  # noqa: F401

"""_summary_"""

import sys
import timeit

import array_tools  # noqa: F401
import numpy as np
import parstack  # noqa: F401
from pyrocko import parstack as pyrocko_parstack  # noqa: F401

sys.path.insert(0, "")


n_nodes = 1000
n_samples = 50000
arr = np.random.uniform(0, 1, (n_nodes, n_samples)).astype(np.float32)
mask = np.random.randint(0, 2, size=n_nodes).astype(bool)
mask = np.ones(n_nodes, dtype=bool)

n_traces = 10

traces = []
for _ in range(n_traces):
    traces.append(np.random.uniform(0, 1, (n_samples)).astype(np.float32))

offsets = np.zeros(n_traces, dtype=np.int32)
shifts = np.random.randint(-100, 100, size=(n_nodes, n_traces)).astype(np.int32)
# shifts = np.zeros((n_nodes, n_traces), dtype=np.int32)
weights = np.ones((n_nodes, n_traces), dtype=np.float32)


def test_argmax():
    indices, max = array_tools.argmax_masked(arr, mask)
    np.testing.assert_array_equal(max, np.max(arr, axis=0))


def test_parstack_mojo():
    res, offset = parstack.parstack(
        traces,
        offsets,
        shifts,
        weights,
        None,
        None,
    )
    return res, offset


def test_parstack_pyrocko():
    res, offset = pyrocko_parstack.parstack(
        traces,
        offsets,
        shifts,
        weights,
        method=0,
        nparallel=1,
    )
    return res, offset


def test_parstack_comp():
    res_mojo, offset_mojo = test_parstack_mojo()
    res_pyrocko, offset_pyrocko = test_parstack_pyrocko()
    assert offset_mojo == offset_pyrocko
    np.testing.assert_allclose(res_mojo, res_pyrocko, rtol=1e-5)


# test_argmax()


def time(timer: timeit.Timer, repeat: int = 5, number: int = 10):
    times = timer.repeat(repeat=repeat, number=number)
    print(f"Min time: {min(times) / number}")
    return min(times) / number


test_parstack_comp()

print("Timeit Mojo:")
timer_mojo = timeit.Timer(lambda: test_parstack_mojo())
time_mojo = time(timer_mojo)
print("Timeit Pyrocko:")
timer_pyrocko = timeit.Timer(lambda: test_parstack_pyrocko())
time_pyrocko = time(timer_pyrocko)
print("Speedup: ", time_pyrocko / time_mojo)
```
