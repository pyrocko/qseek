from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pyrocko.io.mseed import iload, save
from pyrocko.trace import Trace

from qseek.utils import NSL
from qseek.waveforms import sds
from qseek.waveforms.sds import SDSArchive, StationCovarage


@pytest.mark.parametrize("tz", ["Europe/Berlin", "America/Los_Angeles", "Asia/Tokyo"])
def test_available_time_span_is_utc(tz: str) -> None:
    """The time span covers whole UTC days, independent of the local time zone."""
    archive = SDSArchive.model_construct()
    coverage = StationCovarage(
        nsl=NSL("IV", "CPOZ", ""),
        channels={"HHZ"},
        file_dates=[date(2024, 5, 20)],
    )
    archive._archive_stations = {coverage.nsl: coverage}

    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        start, end = archive.available_time_span()
    finally:
        if old_tz is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old_tz
        time.tzset()

    assert start == datetime(2024, 5, 20, tzinfo=timezone.utc)
    assert end == datetime(2024, 5, 21, tzinfo=timezone.utc)


DAY_START = datetime(2024, 5, 20, tzinfo=timezone.utc)


@pytest.fixture
def sds_archive(tmp_path: Path) -> Path:
    """Three hours of two stations in an SDS archive, one channel with a gap."""
    rng = np.random.default_rng(0)
    deltat = 0.01
    tmin = DAY_START.timestamp()
    for station in ("STA", "STB"):
        for channel in ("HHE", "HHN", "HHZ"):
            spans = [(0.0, 10800.0)]
            if (station, channel) == ("STB", "HHZ"):
                spans = [(0.0, 3600.0), (3900.0, 10800.0)]
            traces = [
                Trace(
                    "XX",
                    station,
                    "",
                    channel,
                    tmin=tmin + start,
                    deltat=deltat,
                    ydata=np.cumsum(
                        rng.integers(-500, 500, round((end - start) / deltat))
                    ).astype(np.int32),
                )
                for start, end in spans
            ]
            folder = tmp_path / "2024" / "XX" / station / f"{channel}.D"
            folder.mkdir(parents=True)
            save(traces, str(folder / f"XX.{station}..{channel}.D.2024.141"))
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("n_threads", [0, 2])
async def test_get_traces(sds_archive: Path, n_threads: int) -> None:
    """The files of a window are read and complete traces are kept."""
    archive = SDSArchive(archive=sds_archive)
    archive.scan_sds_archive()
    # As prepare() does, without an executor the default one of asyncio is used
    if n_threads:
        archive._executor = ThreadPoolExecutor(max_workers=n_threads)
    nsls = sorted(archive.available_nsls())
    n_samples = round(420.0 / 0.01)

    start = DAY_START + timedelta(minutes=10)
    traces = await archive.get_traces(nsls, start, start + timedelta(minutes=7))
    assert len(traces) == 6
    for tr in traces:
        assert tr.tmin == start.timestamp()
        assert tr.ydata.size == n_samples
        path = (
            sds_archive
            / "2024"
            / "XX"
            / tr.station
            / f"{tr.channel}.D"
            / f"XX.{tr.station}..{tr.channel}.D.2024.141"
        )
        (expected,) = iload(str(path), tmin=start.timestamp(), tmax=tr.tmax + 0.01)
        np.testing.assert_array_equal(tr.ydata, expected.ydata)

    # Across the gap of STB.HHZ: dropped, or two pieces with incomplete traces
    start = DAY_START + timedelta(minutes=58)
    end = start + timedelta(minutes=10)
    traces = await archive.get_traces(nsls, start, end)
    assert sorted(".".join(tr.nslc_id) for tr in traces) == [
        "XX.STA..HHE",
        "XX.STA..HHN",
        "XX.STA..HHZ",
        "XX.STB..HHE",
        "XX.STB..HHN",
    ]
    traces = await archive.get_traces(nsls, start, end, want_incomplete=True)
    assert len([tr for tr in traces if tr.nslc_id == ("XX", "STB", "", "HHZ")]) == 2


def archive_with_executor(path: Path) -> SDSArchive:
    archive = SDSArchive(archive=path)
    archive.scan_sds_archive()
    archive._executor = ThreadPoolExecutor(max_workers=2)
    return archive


@pytest.mark.asyncio
async def test_get_traces_skips_failing_file(
    sds_archive: Path, monkeypatch, caplog
) -> None:
    """A file that cannot be read is logged, the other files are loaded."""
    archive = archive_with_executor(sds_archive)
    load_file = sds._load_file

    def failing_load_file(file: Path, *args) -> list[Trace]:
        if file.name.startswith("XX.STA..HHE"):
            raise OSError("broken file")
        return load_file(file, *args)

    monkeypatch.setattr(sds, "_load_file", failing_load_file)
    start = DAY_START + timedelta(minutes=10)
    with caplog.at_level("ERROR"):
        traces = await archive.get_traces(
            sorted(archive.available_nsls()), start, start + timedelta(minutes=7)
        )
    assert len(traces) == 5
    assert ("XX", "STA", "", "HHE") not in {tr.nslc_id for tr in traces}
    assert "error loading file: broken file" in caplog.text


@pytest.mark.asyncio
async def test_get_traces_executor_shut_down(sds_archive: Path, caplog) -> None:
    """When the files cannot be loaded at all, the error is logged."""
    archive = archive_with_executor(sds_archive)
    archive._executor.shutdown()
    start = DAY_START + timedelta(minutes=10)
    with caplog.at_level("ERROR"):
        traces = await archive.get_traces(
            sorted(archive.available_nsls()), start, start + timedelta(minutes=7)
        )
    assert traces == []
    assert "error loading files" in caplog.text


@pytest.mark.asyncio
async def test_get_traces_unpadded_day(tmp_path: Path) -> None:
    """Day files with a julian day without zero padding are found."""
    tmin = datetime(2024, 1, 5, tzinfo=timezone.utc)
    trace = Trace(
        "XX",
        "STA",
        "",
        "HHZ",
        tmin=tmin.timestamp(),
        deltat=0.01,
        ydata=np.arange(60000, dtype=np.int32),
    )
    folder = tmp_path / "2024" / "XX" / "STA" / "HHZ.D"
    folder.mkdir(parents=True)
    save([trace], str(folder / "XX.STA..HHZ.D.2024.5"))

    archive = archive_with_executor(tmp_path)
    start = tmin + timedelta(minutes=1)
    traces = await archive.get_traces(
        [NSL("XX", "STA", "")], start, start + timedelta(minutes=2)
    )
    assert [tr.nslc_id for tr in traces] == [("XX", "STA", "", "HHZ")]
    assert traces[0].ydata.size == 12000


def test_station_coverage() -> None:
    """Each station keeps its own channels and file dates."""
    sta = StationCovarage(nsl=NSL("XX", "STA", ""))
    stb = StationCovarage(nsl=NSL("XX", "STB", ""))
    sta.add_file(Path("2024/XX/STA/HHZ.D/XX.STA..HHZ.D.2024.141"))
    stb.add_file(Path("2024/XX/STB/EHZ.D/XX.STB..EHZ.D.2024.142"))
    assert sta.channels == {"HHZ"}
    assert stb.channels == {"EHZ"}
    assert sta.file_dates == [date(2024, 5, 20)]
    assert stb.start_date == date(2024, 5, 21)
    assert not hasattr(sta, "__dict__")
