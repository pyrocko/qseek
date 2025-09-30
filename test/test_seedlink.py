import asyncio
import contextlib
import logging
from datetime import timedelta

import pytest

from qseek.utils import _NSL, datetime_now
from qseek.waveforms.seedlink.client import SeedLinkClient, StationSelection
from qseek.waveforms.seedlink.seedlink import SeedLink, slinktool_available

logging.basicConfig(level=logging.DEBUG)


def get_seedlink_client():
    return SeedLink(
        clients=[
            SeedLinkClient(
                host="geofon.gfz.de",
                port=18000,
                station_selection=[
                    StationSelection(nsl=_NSL("1D", "SYRAU"), channel="HH?"),
                    StationSelection(nsl=_NSL("1D", "WBERG"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "KOC"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "KRC"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "LBC"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "SKC"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "STC"), channel="HH?"),
                    StationSelection(nsl=_NSL("WB", "VAC"), channel="HH?"),
                ],
            )
        ]
    )


async def seedlink_client():
    client = SeedLinkClient(
        host="geofon.gfz-potsdam.de",
        port=18000,
        station_selection=[
            StationSelection(nsl=_NSL("1D", "SYRAU", ""), channel="HH?"),
            StationSelection(nsl=_NSL("1D", "WBERG", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "KOC", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "KRC", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "LBC", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "SKC", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "STC", ""), channel="HH?"),
            StationSelection(nsl=_NSL("WB", "VAC", ""), channel="HH?"),
        ],
    )
    # print(await client.get_available_stations())

    client.start_streams()

    i_batch = 0
    while True:
        await asyncio.sleep(0)
        start = datetime_now()
        traces = []

        traces = await asyncio.gather(
            *[
                stream.get_trace(
                    start_time=start,
                    end_time=start + timedelta(seconds=5),
                    timeout=20.0,
                )
                for stream in client.streams
            ],
            return_exceptions=True,
        )
        assert traces
        i_batch += 1
        if i_batch >= 3:
            break


@pytest.mark.skipif(not slinktool_available(), reason="slinktool not available")
@pytest.mark.asyncio
async def test_seedlink():
    seedlink = get_seedlink_client()

    received_traces = 0

    async def get_batches():
        nonlocal received_traces
        async for batch in seedlink.iter_batches(
            window_increment=timedelta(seconds=5),
            window_padding=timedelta(seconds=5),
            start_time=datetime_now(),
            min_length=timedelta(seconds=5),
            min_stations=1,
        ):
            assert batch.traces
            received_traces += len(batch.traces)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(get_batches(), timeout=30.0)

    assert received_traces > 0


@pytest.mark.skipif(not slinktool_available(), reason="slinktool not available")
@pytest.mark.asyncio
async def test_seedlink_past():
    seedlink = get_seedlink_client()

    received_traces = 0

    async def get_batches():
        nonlocal received_traces

        async for batch in seedlink.iter_batches(
            window_increment=timedelta(seconds=5),
            window_padding=timedelta(seconds=5),
            start_time=datetime_now() - timedelta(days=2),
            min_length=timedelta(seconds=5),
            min_stations=1,
        ):
            assert batch.traces
            received_traces += len(batch.traces)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(get_batches(), timeout=30.0)
    assert received_traces > 0


if __name__ == "__main__":
    asyncio.run(test_seedlink_past())
