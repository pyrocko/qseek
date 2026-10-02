import asyncio
import contextlib
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from qseek.models.station import Station, StationInventory
from qseek.utils import NSL, datetime_now
from qseek.waveforms.seedlink.client import SeedLinkClient, StationSelection
from qseek.waveforms.seedlink.seedlink import SeedLink, slinktool_available

HOST = "geofon.gfz.de"
PORT = 18000
TIMEOUT = 60.0


def seedlink_reachable() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=5.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(not slinktool_available(), reason="slinktool not available"),
    pytest.mark.skipif(not seedlink_reachable(), reason=f"{HOST}:{PORT} unreachable"),
]


@pytest.fixture
def stations() -> StationInventory:
    return StationInventory(
        stations=[
            Station(
                network="GE",
                station="RUE",
                location="",
                lat=52.4759,
                lon=13.78,
                elevation=40.0,
            )
        ]
    )


@pytest_asyncio.fixture
async def seedlink(tmp_path: Path, stations: StationInventory):
    seedlink = SeedLink(
        clients=[
            SeedLinkClient(
                host=HOST,
                port=PORT,
                station_selection=[
                    StationSelection(nsl=NSL("GE", "RUE", ""), channel="HH?"),
                ],
            )
        ],
        sds_archive=tmp_path / "sds",
    )
    await seedlink.prepare(stations)
    assert seedlink.available_nsls() == {NSL("GE", "RUE", "")}
    yield seedlink
    for client in seedlink.clients:
        client.stop_stream()


async def first_batch_traces(seedlink: SeedLink, start_time) -> int:
    """Number of traces in the first streamed batch, 0 if none within the timeout."""

    async def get_first_batch() -> int:
        async for batch in seedlink.iter_batches(
            window_increment=timedelta(seconds=5),
            window_padding=timedelta(seconds=5),
            start_time=start_time,
            min_length=timedelta(seconds=5),
            min_stations=1,
        ):
            assert {tr.station for tr in batch.traces} == {"RUE"}
            return len(batch.traces)
        return 0

    with contextlib.suppress(asyncio.TimeoutError):
        return await asyncio.wait_for(get_first_batch(), timeout=TIMEOUT)
    return 0


@pytest.mark.asyncio
async def test_seedlink(seedlink: SeedLink):
    assert await first_batch_traces(seedlink, start_time=datetime_now()) > 0


@pytest.mark.asyncio
async def test_seedlink_past(seedlink: SeedLink):
    # The server's ring buffer holds several hours of data
    start_time = datetime_now() - timedelta(minutes=10)
    assert await first_batch_traces(seedlink, start_time=start_time) > 0
