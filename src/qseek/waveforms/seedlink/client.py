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


logger = logging.getLogger(__file__)

HOUR = 3600.0
DAY = 24 * HOUR
RECORD_LENGTH = 512


def as_seedlink_time(time: datetime) -> str:
    """Convert a datetime object to a SeedLink time string."""
    # Format: 2002,08,05,14,00,00
    return (
        f"{time.year},{time.month:02d},{time.day:02d},"
        f"{time.hour:02d},{time.minute:02d},{time.second:02d}"
    )


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


class SeedLinkStation(BaseModel):
    network: str
    station: str
    location: str
    channel: str
    dataquality: str

    starttime: datetime
    endtime: datetime

    @classmethod
    def from_line(cls, line: str) -> SeedLinkStation:
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
    nsl: NSL = Field(
        default=NSL("1D", "SYRAU", ""),
        description="Network, station, and location code.",
    )
    channel: str = Field(
        default="???",
        min_length=3,
        max_length=3,
        pattern="[A-Z?]",
    )

    def seedlink_str(self) -> str:
        """Get the SeedLink string for this station."""
        nsl = self.nsl
        return f"{nsl.network}_{nsl.station}:{nsl.location}{self.channel}"


class SeedLinkStream:
    _trace: Trace | None = None
    _last_data: datetime

    _data_received: asyncio.Event

    _fill_rate: float = 0.0

    def __init__(self) -> None:
        self._trace = None
        self._last_data = datetime.min.replace(tzinfo=timezone.utc)
        self._data_received = asyncio.Event()

    def add_trace(self, trace: Trace, max_length_seconds: float = DAY) -> None:
        """Add a trace to the stream.

        Args:
            trace (Trace): Trace to add.
            max_length_seconds (float, optional): Maximum length of the trace in seconds.
                Defaults to 600.0.
        """
        # The day doesn't make sense for historical data when continuing
        # Maybe don't crop traces here but keep partial ones and crop when requested?
        # Or have a max buffer size and drop old data?
        pretty_nslc = ".".join(trace.nslc_id)
        if self._trace is None:
            logger.info("receiving new stream %s", pretty_nslc)
            self._trace = trace

        merged_traces = degapper([self._trace, trace])
        if len(merged_traces) != 1:
            logger.warning("gap in stream %s", pretty_nslc)
        # Only take the latest trace
        # TODO: better take all traces
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

        since_last_data = datetime_now() - self._last_data
        self._fill_rate = (trace.tmax - trace.tmin) / since_last_data.total_seconds()

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

    @property
    def delay(self) -> timedelta:
        """Get the delay of the stream."""
        return datetime_now() - self.end_time

    @property
    def size_bytes(self) -> int:
        """Get the size of the trace in bytes."""
        if not self._trace:
            return 0
        return self._trace.ydata.nbytes

    def is_backfilling(self) -> bool:
        return self._fill_rate > 2.0

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
                logger.warning(
                    "no data from %s within timeout %s s, stream has %.1f s delay",
                    nslc_pretty,
                    timeout,
                    self.delay.total_seconds(),
                )
                raise TimeoutError(
                    f"{nslc_pretty} No data received within timeout"
                ) from exc

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
            logger.debug("no data in the given time window for %s", nslc_pretty)
            raise ValueError(f"{nslc_pretty} has no data for {start_time}") from exc
        return cropped_trace


class SeedLinkClientStats(BaseModel):
    received_bytes: int = Field(
        default=0,
        description="Total number of bytes received from SeedLink.",
    )
    last_packet: datetime = Field(
        default=datetime.min.replace(tzinfo=timezone.utc),
        description="Last time data was received from SeedLink.",
    )
    connected_at: datetime | None = Field(
        default=None,
        description="Time when the SeedLink client was connected.",
    )
    reconnects: int = Field(
        default=0,
        description="Number of times the SeedLink client has reconnected.",
    )

    _seedlink_client: SeedLinkClient | None = PrivateAttr(default=None)
    _timeout: float = PrivateAttr(default=60.0)

    def set_seedlink_client(self, client: SeedLinkClient) -> None:
        self._seedlink_client = client

    def set_timeout(self, timeout: float) -> None:
        self._timeout = timeout

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
        streams = self._seedlink_client.streams
        return len({st.nsl for st in streams if st.is_online(self._timeout)})

    @computed_field
    @property
    def min_delay(self) -> timedelta:
        if not self._seedlink_client:
            return timedelta(seconds=0)
        streams = self._seedlink_client.streams
        if not streams:
            return timedelta(seconds=0)
        return min([st.delay for st in streams])

    @computed_field
    @property
    def host(self) -> str:
        if not self._seedlink_client:
            return "unknown:18000"
        return f"{self._seedlink_client.host}:{self._seedlink_client.port}"

    def is_running(self) -> bool:
        if not self._seedlink_client or not self._seedlink_client._task:
            return False
        return not self._seedlink_client._task.done()

    def add_bytes(self, n_bytes: int) -> None:
        self.received_bytes += n_bytes
        self.last_packet = datetime_now()

    def add_table_row(self, table: Table) -> None:
        host, port = self.host.split(":")
        sign = "↓" if self.is_running() else "[red bold]x[/red bold]"
        table.add_row(
            "Host",
            f"[bold]{host}[/bold]:{port}",
        )
        try:
            table.add_row(
                "Streams",
                f"{sign} {human_readable_bytes(self.received_bytes)}, "
                f"{self.online_stations}/{self.total_stations} online, "
                f"{self.min_delay.total_seconds():.2f} s delay",
            )
        except Exception as exc:
            logger.warning("failed to add table row: %s", exc)


class SeedLinkClient(BaseModel):
    host: str = Field(
        default="geofon.gfz-potsdam.de",
        description="SeedLink server hostname or IP address.",
    )
    port: int = Field(
        default=18000,
        ge=1,
        le=65535,
        description="SeedLink server port.",
    )
    station_selection: list[StationSelection] = Field(
        default=[
            StationSelection(nsl=NSL("1D", "SYRAU", ""), channel="HH?"),
            StationSelection(nsl=NSL("1D", "WBERG", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "KOC", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "KRC", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "LBC", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "SKC", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "STC", ""), channel="HH?"),
            StationSelection(nsl=NSL("WB", "VAC", ""), channel="HH?"),
        ],
        min_length=1,
        description="List of stations to request streams from.",
    )
    buffer_length: timedelta = Field(
        default=timedelta(minutes=30),
        description="Length of the buffer to keep in memory.",
    )
    reconnect_timeout: float = Field(
        default=60.0,
        ge=10.0,
        description="Timeout for reconnecting to the SeedLink server.",
    )

    _stream_data: defaultdict[tuple[str, str, str, str], SeedLinkStream] = PrivateAttr(
        default_factory=lambda: defaultdict(SeedLinkStream)
    )
    _task: asyncio.Task | None = PrivateAttr(default=None)
    _stats: SeedLinkClientStats = PrivateAttr(default_factory=SeedLinkClientStats)

    @property
    def _slink_host(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def streams(self) -> list[SeedLinkStream]:
        if self._task is None:
            raise RuntimeError("Stream is not running")
        return list(self._stream_data.values())

    def get_stats(self) -> SeedLinkClientStats:
        """Get the stats for this client."""
        return self._stats

    def prepare(self, timeout: float = 60.0) -> None:
        self._stats.set_seedlink_client(self)
        self._stats.set_timeout(timeout)

    async def get_available_stations(self) -> list[SeedLinkStation]:
        logger.info("requesting station list from %s", self._slink_host)
        ret = await call_slinktool(["-Q", self._slink_host])
        return [SeedLinkStation.from_line(line.decode()) for line in ret.splitlines()]

    async def stream(
        self,
        start_time: datetime | None = None,
    ) -> None:
        selectors = ",".join(sta.seedlink_str() for sta in self.station_selection)
        nsls = ",".join(sta.nsl.pretty for sta in self.station_selection)
        logger.info("start streaming stations %s from %s", nsls, self._slink_host)

        slinktool_args = [
            "-nt",  # reconnect timeout
            "60",
            "-k",  # heartbeat
            "10",
            "-o",  # output to stdout
            "-",
            "-S",  # selectors
            selectors,
        ]

        if start_time is not None:
            # some SeedLink servers do not support the -tw option and stop delivering
            # data
            logger.info("starting stream at %s", start_time)
            slinktool_args.extend(["-tw", f"{as_seedlink_time(start_time)}:"])

        logger.debug("calling: slinktool %s", " ".join(slinktool_args))
        proc = await asyncio.subprocess.create_subprocess_exec(
            "slinktool",
            *slinktool_args,
            self._slink_host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stats.connected_at = self._stats.connected_at
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        proc.stdout.read(RECORD_LENGTH),
                        timeout=self.reconnect_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "timeout waiting for data from %s, restarting...",
                        self._slink_host,
                    )
                    self._stats.reconnects += 1
                    asyncio.get_running_loop().call_soon(self.restart_stream)
                    break
                self._stats.add_bytes(len(data))

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
                    "received %s.%s.%s.%s from %s",
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
            self._stats.connected_at = None

    def is_backfilliung(self) -> bool:
        return any(sta.is_backfilling() for sta in self.streams)

    def start_streams(self, start_time: datetime | None = None) -> None:
        if self._task is not None:
            raise RuntimeError("Stream is already running")
        self._task = asyncio.create_task(self.stream(start_time=start_time))

    def stop_stream(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        self._stream_data.clear()

    def restart_stream(self) -> None:
        logger.info("restarting streaming %s", self._slink_host)
        self.stop_stream()
        self.start_streams()
