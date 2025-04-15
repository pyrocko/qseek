from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr, computed_field
from pyrocko.io import load
from pyrocko.trace import NoData, Trace, degapper

from qseek.utils import NSL, datetime_now, human_readable_bytes

if TYPE_CHECKING:
    from rich.table import Table

    from qseek.models.station import Stations

logger = logging.getLogger(__file__)

RECORD_LENGTH = 512


def _as_datetime(timestamp: float):
    """Convert a timestamp to a datetime object in UTC."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


async def call_slinktool(cmd_args: list[str]) -> bytes:
    cmd = ["slinktool", *cmd_args]
    logger.debug("Running command: %s", " ".join(cmd))
    proc = await asyncio.subprocess.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ret = await proc.stdout.read(-1)
    await proc.wait()
    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise RuntimeError(
            f"Command failed with code {proc.returncode}: {stderr.decode()}"
        )
    return ret


class SeedLinkStream(BaseModel):
    network: str
    station: str
    location: str
    channel: str
    dataquality: str

    starttime: datetime
    endtime: datetime

    @classmethod
    def from_line(cls, line: str) -> SeedLinkStream:
        # ZB VOSXX 00 HHZ D 2021/10/08 04:20:36.4200  -  2021/10/08 11:31:00.220
        starttime = datetime.strptime(line[18:42], "%Y/%m/%d %H:%M:%S.%f")  # noqa DTZ007
        endtime = datetime.strptime(line[47:71], "%Y/%m/%d %H:%M:%S.%f")  # noqa DTZ007
        starttime = starttime.replace(tzinfo=timezone.utc)
        endtime = endtime.replace(tzinfo=timezone.utc)
        return cls(
            network=line[0:2].strip(),
            station=line[3:8].strip(),
            location=line[9:12].strip(),
            channel=line[12:15].strip(),
            dataquality=line[16],
            starttime=starttime,
            endtime=endtime,
        )


class StationSelection(BaseModel):
    network: str = Field(default="1D", max_length=2)
    station: str = Field(default="SYRAU", max_length=5)
    location: str = Field(default="", max_length=2)
    channel: str = Field(default="???", max_length=3)

    def seedlink_str(self) -> str:
        return f"{self.network}_{self.station}:{self.location}{self.channel}"

    @property
    def nsl(self) -> NSL:
        return NSL(self.network, self.station, self.location)


class SeedLinkData:
    _trace: Trace | None = None
    _last_data: datetime

    _data_received: asyncio.Event

    def __init__(self) -> None:
        self._trace = None
        self._last_data = datetime.min.replace(tzinfo=timezone.utc)
        self._data_received = asyncio.Event()

    def add_trace(self, trace: Trace, max_length_seconds: float = 600.0) -> None:
        """Add a trace to the stream.

        Args:
            trace (Trace): Trace to add.
            max_length_seconds (float, optional): Maximum length of the trace in seconds.
                Defaults to 600.0.
        """
        pretty_nslc = ".".join(trace.nslc_id)
        if self._trace is None:
            logger.info("receiving new stream from %s", pretty_nslc)
            self._trace = trace

        merged_traces = degapper([self._trace, trace])
        if len(merged_traces) != 1:
            logger.warning("gap in trace %s", pretty_nslc)
        # Only take the latest trace
        new_trace = merged_traces[-1]

        if new_trace.tmax - new_trace.tmin > max_length_seconds:
            try:
                new_trace.chop(
                    tmin=new_trace.tmax - max_length_seconds,
                    tmax=new_trace.tmax,
                    inplace=True,
                    want_incomplete=True,
                )
            except NoData:
                logger.warning("failed to chop trace %s", pretty_nslc)
                return
        self._trace = new_trace
        self._last_data = _as_datetime(trace.tmax)

        self._data_received.set()
        self._data_received.clear()

    @property
    def nslc(self) -> tuple[str, str, str, str]:
        if not self._trace:
            return "", "", "", ""
        return self._trace.nslc_id

    @property
    def nsl(self) -> NSL:
        if not self._trace:
            return NSL("", "", "")
        return NSL(self._trace.network, self._trace.station, self._trace.location)

    @property
    def start_time(self) -> datetime:
        if not self._trace:
            return datetime.min.replace(tzinfo=timezone.utc)
        return _as_datetime(self._trace.tmin)

    @property
    def end_time(self) -> datetime:
        if not self._trace:
            return datetime.min.replace(tzinfo=timezone.utc)
        return _as_datetime(self._trace.tmax)

    @property
    def length(self) -> timedelta:
        return self.end_time - self.start_time

    def is_online(self, timeout: float = 60.0) -> bool:
        """Check if the stream is online in the seconds.

        Args:
            timeout: Timeout in seconds.
        """
        return self._last_data > datetime.now(timezone.utc) - timedelta(seconds=timeout)

    async def get_trace(
        self,
        start_time: datetime,
        end_time: datetime,
        timeout: float = 20.0,
    ) -> Trace:
        """Wait for data in the given time window.

        Args:
            start_time: Start time of the time window.
            end_time: End time of the time window.
            timeout: Timeout in seconds.

        Raises:
            ValueError: If the end time is not greater than the start time.
            TimeoutError: If no data is received in the given time window.
        """
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        if self.start_time < start_time:
            try:
                logger.debug("waiting for first data...")
                await asyncio.wait_for(self._data_received.wait(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"No data received before {start_time}") from exc

        timeout_time = end_time + timedelta(seconds=timeout)
        nslc_pretty = ".".join(self.nslc)
        while self.end_time < end_time:
            try:
                await asyncio.wait_for(
                    self._data_received.wait(),
                    timeout=(timeout_time - datetime_now()).total_seconds(),
                )
            except asyncio.TimeoutError as exc:
                logger.warning("no data received for %s", nslc_pretty)
                raise TimeoutError("No data received in the given time window") from exc

        if self._trace is None:
            raise ValueError("No trace available")

        try:
            cropped_trace = self._trace.chop(
                tmin=start_time.timestamp(),
                tmax=end_time.timestamp(),
                want_incomplete=False,
                inplace=False,
            )
        except NoData as exc:
            logger.warning("no data in the given time window for %s", nslc_pretty)
            raise ValueError(
                f"{nslc_pretty} No data available in the given time window"
            ) from exc
        return cropped_trace


class SeedLinkClientStats(BaseModel):
    received_bytes: int = Field(
        default=0,
        description="Total number of bytes received from SeedLink.",
    )
    last_data: datetime = Field(
        default=datetime.min.replace(tzinfo=timezone.utc),
        description="Last time data was received from SeedLink.",
    )

    _seedlink_client: SeedLinkClient | None = PrivateAttr(default=None)

    def set_seedlink_client(self, client: SeedLinkClient) -> None:
        self._seedlink_client = client

    @computed_field
    @property
    def total_streams(self) -> int:
        if not self._seedlink_client:
            return 0
        return len(self._seedlink_client.streams)

    @computed_field
    @property
    def online_streams(self) -> int:
        if not self._seedlink_client:
            return 0
        return sum(1 for sta in self._seedlink_client.streams if sta.is_online())

    @computed_field
    @property
    def total_stations(self) -> int:
        if not self._seedlink_client:
            return 0
        return len(self._seedlink_client.station_selection)

    @computed_field
    @property
    def online_stations(self) -> int:
        if not self._seedlink_client:
            return 0
        return len({st.nsl for st in self._seedlink_client.streams if st.is_online()})

    @computed_field
    @property
    def host(self) -> str:
        if not self._seedlink_client:
            return ""
        return f"{self._seedlink_client.host}:{self._seedlink_client.port}"

    def add_table_row(self, table: Table) -> None:
        age_data = datetime_now() - self.last_data
        table.add_row(
            "Host",
            f"{self.host}",
        )
        table.add_row(
            "Streams",
            f"{human_readable_bytes(self.received_bytes)} rcvd, "
            f"{self.online_stations}/{self.total_stations} online, "
            f"{age_data.total_seconds():.0f}s since last data",
        )


class SeedLinkClient(BaseModel):
    host: str = Field(
        default="geofon.gfz-potsdam.de",
        description="SeedLink shostname.",
    )
    port: int = Field(
        default=18000,
        ge=1,
        le=65535,
        description="SeedLink server port.",
    )
    station_selection: list[StationSelection] = Field(
        default=[
            StationSelection(network="1D", station="SYRAU", location="", channel="HH?"),
            StationSelection(network="1D", station="WBERG", location="", channel="HH?"),
            StationSelection(network="WB", station="KOC", location="", channel="HH?"),
            StationSelection(network="WB", station="KRC", location="", channel="HH?"),
            StationSelection(network="WB", station="LBC", location="", channel="HH?"),
            StationSelection(network="WB", station="SKC", location="", channel="HH?"),
            StationSelection(network="WB", station="STC", location="", channel="HH?"),
            StationSelection(network="WB", station="VAC", location="", channel="HH?"),
        ],
        description="List of stations to request streams from.",
    )
    buffer_length: timedelta = Field(
        default=timedelta(minutes=10),
        description="Length of the buffer to keep in memory.",
    )

    _stream_data: defaultdict[tuple[str, str, str, str], SeedLinkData] = PrivateAttr(
        default_factory=lambda: defaultdict(SeedLinkData)
    )
    _task: asyncio.Task | None = PrivateAttr(default=None)
    _stats: SeedLinkClientStats = PrivateAttr(default_factory=SeedLinkClientStats)

    @property
    def _slink_host(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def streams(self) -> list[SeedLinkData]:
        if self._task is None:
            raise RuntimeError("Stream is not running")
        return list(self._stream_data.values())

    def get_stats(self) -> SeedLinkClientStats:
        """Get the stats for this client."""
        return self._stats

    def prepare(self, stations: Stations) -> None:
        self._stats.set_seedlink_client(self)
        self._stations = stations

    async def get_available_stations(self) -> list[SeedLinkStream]:
        logger.info("requesting station list from %s", self._slink_host)
        ret = await call_slinktool(["-Q", self._slink_host])
        return [SeedLinkStream.from_line(line.decode()) for line in ret.splitlines()]

    async def stream(self) -> None:
        selectors = ",".join(sta.seedlink_str() for sta in self.station_selection)
        logger.info("start streaming stations %s from %s", selectors, self._slink_host)

        proc = await asyncio.subprocess.create_subprocess_exec(
            "slinktool",
            "-o",
            "-",
            "-S",
            selectors,
            self._slink_host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                data = await proc.stdout.read(RECORD_LENGTH)
                self._stats.last_data = datetime_now()
                self._stats.received_bytes += len(data)
                with NamedTemporaryFile() as tmpfile:
                    tmpfile.write(data)
                    tmpfile.flush()
                    try:
                        traces = load(tmpfile.name)
                    except Exception as exc:
                        logger.warning("failed to load trace: %s", exc)
                        continue

                if len(traces) != 1:
                    logger.error("expected 1 trace, got %d", len(traces))
                    continue

                trace: Trace = traces[0]
                logger.debug(
                    "received stream %s.%s.%s.%s from %s",
                    *trace.nslc_id,
                    self._slink_host,
                )
                station_data = self._stream_data[trace.nslc_id]
                station_data.add_trace(
                    trace,
                    max_length_seconds=self.buffer_length.total_seconds(),
                )
        finally:
            proc.terminate()
            raise

    def start_streams(self) -> None:
        if self._task is not None:
            raise RuntimeError("Stream is already running")
        self._task = asyncio.create_task(self.stream())

    def stop_stream(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        self._stream_data.clear()
