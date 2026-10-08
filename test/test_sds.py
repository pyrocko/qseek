from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pyrocko.io.mseed import iload, save
from pyrocko.trace import Trace

from qseek.utils import NSL
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
async def test_get_traces(sds_archive: Path) -> None:
    """The files of a window are read and complete traces are kept."""
    archive = SDSArchive(archive=sds_archive)
    archive.scan_sds_archive()
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
