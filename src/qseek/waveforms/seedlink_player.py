from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import AwareDatetime, BaseModel, DirectoryPath, Field
from pyrocko.io.mseed import get_bytes
from pyrocko.squirrel import Squirrel

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch
    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)


RECORD_LENGTH = 512


class SeedLinkStation:
    data: dict[str, asyncio.Queue[bytes]]
    _task: asyncio.Task | None = None

    def __init__(self, network: str, station: str, speed: float, fifo: Path) -> None:
        self.network = network
        self.station = station
        self.speed = speed
        self.fifo = fifo
        self.delay = 0.0
        self._data = defaultdict(lambda: asyncio.Queue(maxsize=512))

    @classmethod
    def from_traces(cls, traces: list[Trace]) -> Self:
        network = {trace.network for trace in traces}
        station = {trace.station for trace in traces}
        if len(network) != 1 or len(station) != 1:
            raise ValueError("Traces must have the same network and station")
        return cls(network.pop(), station.pop())

    def add_traces(self, traces: list[Trace]) -> None:
        for trace in traces:
            if trace.network != self.network or trace.station != self.station:
                raise ValueError("Traces must have the same network and station")

        for cha, cha_traces in groupby(traces, key=lambda trace: trace.channel):
            cha_traces = list(cha_traces)
            tmin = min(trace.tmin for trace in cha_traces)
            tmax = max(trace.tmax for trace in cha_traces)
            duration = tmax - tmin

            data = get_bytes(cha_traces, record_length=RECORD_LENGTH, steim=2)
            n_records = len(data) // RECORD_LENGTH
            self.delay = n_records / duration / self.speed

            for irec in range(n_records):
                while True:
                    try:
                        self._data[cha].put_nowait(
                            data[irec * RECORD_LENGTH : (irec + 1) * RECORD_LENGTH]
                        )
                        break
                    except asyncio.QueueFull:
                        logger.warning("Queue full, dropping data")
                        self._data[cha].get_nowait()

    def size_bytes(self) -> int:
        return sum(queue.qsize() * RECORD_LENGTH for queue in self._data.values())

    def start(self) -> None:
        if self.is_streaming():
            return
        logger.info("Starting SeedLink station %s.%s", self.network, self.station)
        self._task = asyncio.create_task(self.stream())

    def stop(self) -> None:
        if self._task is None:
            return
        logger.info("Stopping SeedLink station %s.%s", self.network, self.station)
        self._task.cancel()
        self._task = None

    def is_streaming(self) -> bool:
        if self._task is None:
            return False
        return self._task is not None and not self._task.done()

    async def drain(self) -> None:
        while self.size_bytes() > 0:
            await asyncio.sleep(0.1)
        self.stop()

    async def stream(self) -> bytes:
        while True:
            for queue in self._data.values():
                data = await queue.get()
                try:
                    self.fifo.write_bytes(data)
                except FileNotFoundError:
                    logger.error("Cannot write to %s", self.fifo)
            await asyncio.sleep(self.delay)


class SeedLinkPlayer(BaseModel):
    squirrel_environment: DirectoryPath = Field(
        default=DirectoryPath("."),
        description="Path to the squirrel environment directory",
    )
    squirrel_persistent: str = Field(
        default="squirrel.sqlite",
        description="Name of the squirrel persistent selection",
    )
    start_time: AwareDatetime | None = Field(
        default=None,
        description="Start time of the data to play",
    )
    end_time: AwareDatetime | None = Field(
        default=None,
        description="Start time of the data to play",
    )
    seedlink_fifo: Path = Field(
        default=Path("/tmp/seedlink_player.mseed"),
        description="Path to the SeedLink FIFO file",
    )
    increment_seconds: float = Field(
        default=60.0,
        description="Increment in seconds for the data playback",
    )
    speed: float = Field(
        default=10.0,
        description="Speed of the data playback",
    )

    _squirrel: Squirrel | None = None
    _stations: dict[tuple[str, str], SeedLinkStation] = {}

    def get_squirrel(self) -> Squirrel:
        if self._squirrel is None:
            logger.info("Initializing squirrel")
            self._squirrel = Squirrel(
                self.squirrel_environment,
                persistent=self.squirrel_persistent,
            )
        return self._squirrel

    async def start_seedlink(self): ...

    async def stop_seedlink(self): ...

    def get_station(self, network: str, station: str) -> SeedLinkStation:
        key = (network, station)
        if key not in self._stations:
            sl_station = SeedLinkStation(
                network,
                station,
                speed=self.speed,
                fifo=self.seedlink_fifo,
            )
            sl_station.start()
            self._stations[key] = sl_station

        return self._stations[key]

    def add_batch(self, batch: Batch) -> None:
        for (network, station), traces in groupby(
            batch.traces, key=lambda tr: (tr.network, tr.station)
        ):
            station = self.get_station(network, station)
            station.add_traces(list(traces))

    async def drain_all(self) -> None:
        logger.info("Draining all stations")
        await asyncio.gather(*(station.drain() for station in self._stations.values()))

    async def feed_waveforms(self):
        logging.info("Starting playback of waveforms with speed %f", self.speed)

        squirrel = self.get_squirrel()
        iterator = squirrel.chopper_waveforms(
            tmin=self.start_time.timestamp() if self.start_time else None,
            tmax=self.end_time.timestamp() if self.end_time else None,
            want_incomplete=True,
            tinc=self.increment_seconds,
        )

        for iiter in range(10):
            logging.info("Loading batch %d", iiter)
            batch = await asyncio.to_thread(next, iterator, None)
            if batch is None:
                break
            self.add_batch(batch)
            await asyncio.sleep(self.increment_seconds / self.speed)

        await self.drain_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    player = SeedLinkPlayer(
        squirrel_environment="",
        squirrel_persistent="la_palma",
        seedlink_fifo="/tmp/mseed_fifo",
        speed=10.0,
    )

    asyncio.run(player.feed_waveforms())
