from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pyrocko.io.mseed import save
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
async def test_mseed_loaders(sds_archive: Path, monkeypatch) -> None:
    """Both MiniSEED loaders return the same traces."""
    archive = SDSArchive(archive=sds_archive)
    archive.scan_sds_archive()
    nsls = sorted(archive.available_nsls())

    windows = [
        (DAY_START - timedelta(minutes=1), timedelta(minutes=7)),
        (DAY_START + timedelta(minutes=55), timedelta(minutes=7)),
        (DAY_START + timedelta(minutes=58, seconds=0.005), timedelta(minutes=10)),
        (DAY_START + timedelta(hours=2, minutes=55), timedelta(minutes=7)),
    ]
    results = {}
    for loader in ("qseek", "pyrocko"):
        monkeypatch.setattr(sds, "MSEED_LOADER", loader)
        results[loader] = []
        for start, length in windows:
            for want_incomplete in (True, False):
                traces = await archive.get_traces(
                    nsls, start, start + length, want_incomplete=want_incomplete
                )
                results[loader].append(
                    sorted(
                        (tr.nslc_id, tr.tmin, tr.tmax, tr.ydata.tobytes())
                        for tr in traces
                    )
                )
    assert results["qseek"] == results["pyrocko"]
    assert any(results["qseek"])


@pytest.mark.asyncio
async def test_unknown_mseed_loader(sds_archive: Path, monkeypatch, caplog) -> None:
    archive = SDSArchive(archive=sds_archive)
    archive.scan_sds_archive()
    monkeypatch.setattr(sds, "MSEED_LOADER", "obspy")
    start = DAY_START + timedelta(hours=1)
    with caplog.at_level("ERROR"):
        traces = await archive.get_traces(
            sorted(archive.available_nsls()), start, start + timedelta(minutes=5)
        )
    assert traces == []
    assert "unknown MiniSEED loader 'obspy'" in caplog.text
