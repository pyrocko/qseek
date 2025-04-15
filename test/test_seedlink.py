import asyncio
import logging
from datetime import timedelta

from qseek.utils import datetime_now
from qseek.waveforms.seedlink.client import SeedLinkClient, StationSelection
from qseek.waveforms.seedlink.seedlink import SeedLink

logging.basicConfig(level=logging.DEBUG)


async def seedlink_client():
    client = SeedLinkClient(
        host="geofon.gfz-potsdam.de",
        port=18000,
        station_selection=[
            StationSelection(network="1D", station="SYRAU", location="", channel="HH?"),
            StationSelection(network="1D", station="WBERG", location="", channel="HH?"),
            StationSelection(network="WB", station="KOC", location="", channel="HH?"),
            StationSelection(network="WB", station="KRC", location="", channel="HH?"),
            StationSelection(network="WB", station="LBC", location="", channel="HH?"),
            StationSelection(network="WB", station="SKC", location="", channel="HH?"),
            StationSelection(network="WB", station="STC", location="", channel="HH?"),
            StationSelection(network="WB", station="VAC", location="", channel="HH?"),
        ],
    )
    # print(await client.get_available_stations())

    client.start_streams()

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


async def seedlink():
    seedlink = SeedLink(
        clients=[
            SeedLinkClient(
                host="geofon.gfz-potsdam.de",
                port=18000,
                station_selection=[
                    StationSelection(
                        network="1D", station="SYRAU", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="1D", station="WBERG", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="KOC", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="KRC", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="LBC", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="SKC", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="STC", location="", channel="HH?"
                    ),
                    StationSelection(
                        network="WB", station="VAC", location="", channel="HH?"
                    ),
                ],
            )
        ]
    )
    # print(await client.get_available_stations())

    async for batch in seedlink.iter_batches(
        window_increment=timedelta(seconds=5),
        window_padding=timedelta(seconds=5),
        start_time=datetime_now(),
        min_length=timedelta(seconds=5),
        min_stations=1,
    ):
        assert batch.traces


asyncio.run(seedlink())
