from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone

import pytest

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
