from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal

from pydantic import Field, PrivateAttr, computed_field
from pyrocko.io import save
from pyrocko.squirrel import Squirrel
from pyrocko.trace import Trace
from rich.table import Table

from qseek.stats import Stats
from qseek.utils import datetime_now
from qseek.waveforms.base import WaveformBatch, WaveformProvider
from qseek.waveforms.seedlink.client import SeedLinkClient, SeedLinkClientStats

if TYPE_CHECKING:
    from qseek.models.station import Stations

logger = logging.getLogger(__name__)

SDS_TEMPLATE = (
    "%(tmin_year)s/%(network)s/%(station)s/%(channel)s.D"
    "/%(network)s.%(station)s.%(location)s.%(channel)s.D"
    ".%(tmin_year)s.%(julianday)s"
)


async def save_traces_sds(
    traces: list[Trace],
    sds_directory: Path,
    crop_padding: timedelta,
) -> list[Path]:
    """Save traces in SDS format.

    Args:
        traces: List of traces to save.
        sds_directory: Directory to save the traces in.
        crop_padding: Padding to crop the traces.

    Returns:
        List of filenames of the saved traces.
    """
    if not sds_directory.exists():
        sds_directory.mkdir(parents=True)
    if not sds_directory.is_dir():
        raise ValueError(f"{sds_directory} is not a directory")

    saved_filenames = []
    for trace in traces:
        trace = trace.chop(
            tmin=trace.tmin + crop_padding.total_seconds(),
            tmax=trace.tmax - crop_padding.total_seconds(),
            want_incomplete=True,
            inplace=False,
        )
        filename = str(sds_directory / SDS_TEMPLATE)
        tr_tmin = datetime.fromtimestamp(trace.tmin, tz=timezone.utc)
        tr_julianday = (tr_tmin + timedelta(seconds=10.0)).timetuple().tm_yday
        fns = await asyncio.to_thread(
            save,
            [trace],
            filename_template=filename,
            steim=2,
            record_length=4096,
            additional={"julianday": tr_julianday},
            append=True,
        )
        saved_filenames.extend(tuple(map(Path, fns)))
    return saved_filenames


class SeedLinkStats(Stats):
    """Statistics for SeedLink streams."""

    _seedlink: SeedLink | None = PrivateAttr(default=None)

    def set_seedlink(self, seedlink: SeedLink) -> None:
        """Set the SeedLink instance for statistics."""
        self._seedlink = seedlink

    @computed_field
    @property
    def clients(self) -> list[SeedLinkClientStats]:
        if self._seedlink is None:
            return []
        return [client.get_stats() for client in self._seedlink.clients]

    @computed_field
    @property
    def total_bytes(self) -> int:
        return sum(client.received_bytes for client in self.clients)

    def _populate_table(self, table: Table) -> None:
        for client in self.clients:
            client.add_table_row(table)


class SeedLink(WaveformProvider):
    """Waveform provider for SeedLink real-time streams."""

    provider: Literal["SeedLink"] = "SeedLink"
    timeout: timedelta = Field(
        default=timedelta(seconds=20),
        description="Maximum wait time for new traces.",
    )

    clients: list[SeedLinkClient] = Field(default=[SeedLinkClient()])
    save_sds_archive: Path = Field(
        default=Path("./sds-seedlink"),
        description="Path to save MiniSeed in an SDS structure. Give a path to save "
        "the archive, or True to use the default path. If False, no saving is done.",
    )

    _stats: SeedLinkStats = PrivateAttr(default_factory=SeedLinkStats)
    _squirrel: Squirrel = PrivateAttr(None)
    _saved_filenames: set[Path] = PrivateAttr(default_factory=set)

    def prepare(self, stations: Stations) -> None:
        """Prepare the SeedLink client."""
        self._stats.set_seedlink(self)
        self.save_sds_archive.mkdir(parents=True, exist_ok=True)

        self._squirrel = Squirrel()
        self._squirrel.add(str(self.save_sds_archive), check=False)

        streaming_nsls = {
            station.nsl
            for client in self.clients
            for station in client.station_selection
        }
        stations.weed_from_nsls(streaming_nsls)

        for client in self.clients:
            client.prepare(stations)

        for client in self.clients:
            client.start_streams()

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> AsyncIterator[WaveformBatch]:
        start_time = datetime_now()
        i_batch = 0

        while True:
            await asyncio.sleep(0)
            start_time_padded = start_time - window_padding
            end_time_padded = start_time + window_increment + window_padding
            logger.info("streaming traces to %s", end_time_padded)
            try:
                seedlink_traces = await asyncio.gather(
                    *(
                        stream.get_trace(
                            start_time=start_time_padded,
                            end_time=end_time_padded,
                            timeout=self.timeout.total_seconds(),
                        )
                        for client in self.clients
                        for stream in client.streams
                    ),
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.warning("failed to get traces: %s", exc)
                start_time += window_increment
                continue
            batch_traces = [tr for tr in seedlink_traces if isinstance(tr, Trace)]
            if not batch_traces:
                logger.warning("no traces received")
                start_time += timedelta(seconds=5.0)
                await asyncio.sleep(5.0)
                continue

            logger.debug("received %d traces", len(batch_traces))
            filenames = await save_traces_sds(
                traces=batch_traces,
                sds_directory=self.save_sds_archive,
                crop_padding=window_padding,
            )
            self._saved_filenames.update(filenames)
            self._squirrel.add(map(str, filenames), check=False)

            i_batch += 1
            batch = WaveformBatch(
                traces=batch_traces,
                start_time=start_time,
                end_time=start_time + window_increment,
                i_batch=i_batch,
            )

            batch.clean_traces()
            if not batch.is_healthy(min_stations=min_stations):
                logger.warning("batch is not healthy, skipping")
                start_time += window_increment
                continue
            if min_length and batch.duration < min_length:
                logger.warning(
                    "duration of batch %d too short %s",
                    batch.i_batch,
                    batch.duration,
                )
                continue

            yield batch
            start_time += window_increment

    def get_squirrel(self) -> Squirrel:
        """Get the Squirrel instance."""
        return self._squirrel
