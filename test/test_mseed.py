from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pyrocko.io.mseed import iload, save
from pyrocko.trace import Trace

from qseek.waveforms import mseed
from qseek.waveforms.mseed import get_layout, load_time_window

TMIN = 1716163200.0  # 2024-05-20
DELTAT = 0.01


def make_trace(tmin: float, n_samples: int, channel: str = "HHZ") -> Trace:
    rng = np.random.default_rng(int(tmin) % 1000)
    data = np.cumsum(rng.integers(-500, 500, n_samples)).astype(np.int32)
    return Trace("XX", "STA", "", channel, tmin=tmin, deltat=DELTAT, ydata=data)


def write(path: Path, traces: list[Trace], record_length: int = 4096) -> Path:
    save(traces, str(path), record_length=record_length, steim=2)
    return path


def windows(tmin: float, tmax: float) -> list[tuple[float, float]]:
    rng = np.random.default_rng(42)
    fixed = [
        (tmin - 100.0, tmin + 50.0),
        (tmax - 30.0, tmax + 100.0),
        (tmin, tmin + 300.0),
        (tmin - 200.0, tmin - 100.0),
        (tmax + 100.0, tmax + 200.0),
    ]
    random = []
    for _ in range(40):
        start = float(rng.uniform(tmin - 50.0, tmax))
        if rng.random() < 0.5:
            start = round(start)
        random.append((start, start + float(rng.choice([1.0, 40.0, 420.0]))))
    return fixed + random


def assert_same_traces(path: Path, tmin: float, tmax: float) -> None:
    expected = [
        tr
        for tr in iload(str(path), tmin=tmin, tmax=tmax)
        # iload returns traces without data in the window unchopped
        if tr.tmin >= tmin - tr.deltat and tr.tmax < tmax + tr.deltat
    ]
    traces = load_time_window(path, tmin, tmax)
    assert len(traces) == len(expected)
    for tr, tr_expected in zip(traces, expected, strict=True):
        assert tr.nslc_id == tr_expected.nslc_id
        assert tr.tmin == tr_expected.tmin
        assert tr.tmax == tr_expected.tmax
        assert tr.deltat == tr_expected.deltat
        assert tr.ydata.dtype == tr_expected.ydata.dtype
        np.testing.assert_array_equal(tr.ydata, tr_expected.ydata)


@pytest.mark.parametrize("record_length", [512, 4096])
def test_load_time_window(tmp_path: Path, record_length: int) -> None:
    path = write(tmp_path / "data.mseed", [make_trace(TMIN, 360_000)], record_length)
    layout = get_layout(path)
    assert layout is not None
    assert layout.record_length == record_length

    for tmin, tmax in windows(TMIN, TMIN + 3600.0):
        assert_same_traces(path, tmin, tmax)


def test_load_time_window_record_boundaries(tmp_path: Path) -> None:
    record_length = 512
    path = write(
        tmp_path / "data.mseed", [make_trace(TMIN + 0.123, 100_000)], record_length
    )
    layout = get_layout(path)
    assert layout is not None

    for i_record in range(0, layout.n_records, 7):
        (record,) = iload(
            str(path),
            offset=i_record * record_length,
            segment_size=record_length,
            nsegments=1,
        )
        for tmin in (
            record.tmin,
            record.tmax,
            record.tmax + 0.5 * DELTAT,
            record.tmax + DELTAT,
        ):
            assert_same_traces(path, tmin, tmin + 3.0)
            assert_same_traces(path, tmin - 3.0, tmin)


def test_load_time_window_gaps(tmp_path: Path) -> None:
    traces = [
        make_trace(TMIN, 60_000),
        make_trace(TMIN + 700.0, 30_000),
        make_trace(TMIN + 1000.005, 60_000),
    ]
    path = write(tmp_path / "data.mseed", traces)
    assert get_layout(path) is not None

    for tmin, tmax in windows(TMIN, TMIN + 1600.0):
        assert_same_traces(path, tmin, tmax)


def test_load_time_window_out_of_order(tmp_path: Path) -> None:
    first = write(tmp_path / "first.mseed", [make_trace(TMIN, 60_000)])
    second = write(tmp_path / "second.mseed", [make_trace(TMIN + 600.0, 60_000)])
    path = tmp_path / "data.mseed"
    path.write_bytes(second.read_bytes() + first.read_bytes())
    assert get_layout(path) is None

    for tmin, tmax in windows(TMIN, TMIN + 1200.0):
        assert_same_traces(path, tmin, tmax)


def test_load_time_window_overlap(tmp_path: Path) -> None:
    single = write(tmp_path / "single.mseed", [make_trace(TMIN, 60_000)])
    path = tmp_path / "data.mseed"
    path.write_bytes(single.read_bytes() * 2)
    assert get_layout(path) is None


def test_load_time_window_channels(tmp_path: Path) -> None:
    traces = [make_trace(TMIN, 60_000, "HHZ"), make_trace(TMIN, 60_000, "HHN")]
    path = write(tmp_path / "data.mseed", traces)
    assert get_layout(path) is None

    for tmin, tmax in windows(TMIN, TMIN + 600.0):
        assert_same_traces(path, tmin, tmax)


def test_load_time_window_growing_file(tmp_path: Path) -> None:
    path = write(tmp_path / "data.mseed", [make_trace(TMIN, 60_000)])
    assert_same_traces(path, TMIN + 500.0, TMIN + 700.0)

    more = write(tmp_path / "more.mseed", [make_trace(TMIN + 600.0, 60_000)])
    with path.open("ab") as file:
        file.write(more.read_bytes())

    assert_same_traces(path, TMIN + 500.0, TMIN + 700.0)
    assert load_time_window(path, TMIN + 1100.0, TMIN + 1150.0)


@pytest.fixture(scope="module")
def day_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A day file of an SDS archive: 24 h at 100 Hz in records of 4096 bytes."""
    path = tmp_path_factory.mktemp("sds") / "XX.STA..HHZ.D.2024.141"
    return write(path, [make_trace(TMIN, int(86400 / DELTAT))])


@pytest.mark.benchmark(group="mseed_time_window")
@pytest.mark.parametrize("loader", ["pyrocko", "qseek", "qseek-uncached"])
def test_load_time_window_benchmark(
    benchmark,
    day_file: Path,
    loader: Literal["pyrocko", "qseek", "qseek-uncached"],
) -> None:
    # A batch of a search: 5 min and 1 min padding on both sides, at midday
    tmin = TMIN + 43200.0
    tmax = tmin + 420.0

    def load_pyrocko() -> list[Trace]:
        return list(iload(str(day_file), tmin=tmin, tmax=tmax))

    def load_qseek() -> list[Trace]:
        return load_time_window(day_file, tmin, tmax)

    if loader == "pyrocko":
        traces = benchmark(load_pyrocko)
    elif loader == "qseek":
        load_qseek()
        traces = benchmark(load_qseek)
    else:
        # The first read of a file indexes its records
        traces = benchmark.pedantic(
            load_qseek,
            setup=mseed._record_layout.cache_clear,
            rounds=20,
        )

    (trace,) = traces
    assert trace.tmin == tmin
    assert trace.ydata.size == round((tmax - tmin) / DELTAT)
