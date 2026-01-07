from random import choices
from typing import Literal

import numpy as np
import pytest

from qseek.utils import NSL


@pytest.mark.benchmark(group="dict_lookup")
@pytest.mark.parametrize("implementation", ["NSL", "str"])
def test_dict_lookup(implementation: Literal["NSL", "str"], benchmark):
    n_stations = 1000
    table_nsl: dict[NSL, int] = {NSL("X", f"{i:05d}", ""): i for i in range(n_stations)}
    table_str: dict[str, int] = {nsl.pretty: i for nsl, i in table_nsl.items()}

    nsls = list(table_nsl.keys())
    nsls_pretty = [nsl.pretty for nsl in nsls]

    def lookup_nsl():
        for nsl in nsls:
            _ = table_nsl[nsl]

    def lookup_str():
        for nsl in nsls_pretty:
            _ = table_str[nsl]

    if implementation == "NSL":
        benchmark(lookup_nsl)
    else:
        benchmark(lookup_str)


@pytest.mark.benchmark(group="array_indexing")
@pytest.mark.parametrize("index", ["list", "array"])
def test_numpy_index(benchmark, index: Literal["list", "array"]):
    n = 100000

    data = np.arange(n)
    indices = np.random.randint(0, n, size=n // 10)
    indices_list = indices.tolist()

    def index_list():
        _ = data[indices_list]

    def index_array():
        _ = data[indices]

    if index == "list":
        benchmark(index_list)
    else:
        benchmark(index_array)


@pytest.mark.benchmark(group="list_indexing")
@pytest.mark.parametrize("index", ["list", "dict"])
def test_list_index(benchmark, index: Literal["list", "dict"]):
    n = 100

    stations = list(range(n))
    index_dict = {f"S{i:03d}": i for i in stations}
    indices_list = list(index_dict.keys())

    station_choices = choices(indices_list, k=n // 10)

    def index_list():
        return [indices_list.index(i) for i in station_choices]

    def index_dict_lookup():
        return [index_dict[i] for i in station_choices]

    if index == "list":
        benchmark(index_list)
    else:
        benchmark(index_dict_lookup)


@pytest.mark.benchmark(group="list_indexing")
@pytest.mark.parametrize("index", ["loop", "comprehension"])
def test_lut_filler(benchmark, index: Literal["loop", "comprehension"]):
    n_entries = 50000

    stations = list(range(n_entries))
    index_dict = {f"S{i:05d}": i for i in stations}

    station_choices = choices(list(index_dict.keys()), k=10000)

    def fill_loop():
        res = []
        for i in station_choices:
            try:
                res.append(index_dict[i])
            except KeyError:
                raise
        return np.array(res)

    def fill_comprehension():
        try:
            res = [index_dict[i] for i in station_choices]
            return np.array(res)
        except KeyError:
            raise

    if index == "loop":
        benchmark(fill_loop)
    else:
        benchmark(fill_comprehension)
