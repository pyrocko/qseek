from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal, NamedTuple

from obspy import UTCDateTime, read
from pydantic import (
    AwareDatetime,
    DirectoryPath,
    Field,
    PositiveInt,
    PrivateAttr,
    model_validator,
)
from pyrocko import obspy_compat
from pyrocko.trace import degapper
from typing_extensions import Self

from qseek.stats import Stats, get_progress
from qseek.utils import NSL
from qseek.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.station import Stations


logger = logging.getLogger(__name__)


async def _load_file(
    file: Path,
    start_time: datetime,
    end_time: datetime,
    executor: ThreadPoolExecutor | None = None,
) -> list[Trace]:
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = partial(
        ctx.run,
        read,
        str(file),
        starttime=UTCDateTime(start_time),
        endtime=UTCDateTime(end_time),
    )
    stream = await loop.run_in_executor(executor, func_call)
    return stream.to_pyrocko_traces()


async def _load_files(
    files: list[Path],
    start_time: datetime,
    end_time: datetime,
    executor: ThreadPoolExecutor | None = None,
) -> list[Trace]:
    try:
        traces = await asyncio.gather(
            *[
                _load_file(file, start_time, end_time, executor=executor)
                for file in files
            ],
            return_exceptions=True,
        )
    except Exception as e:
        logger.error("error loading files: %s", e)
        return []

    traces = list(chain(*traces))
    return degapper(sorted(traces, key=lambda tr: tr.full_id))


def nsl_from_path(file: str) -> NSL:
    network, station, location, *_ = file.split(".")
    return NSL(network, station, location)


class StationCovarage(NamedTuple):
    nsl: NSL
    channels: set[str] = set()
    file_dates: list[date] = []

    def add_file(self, file: Path):
        *_, network, station, channel, filename = file.parts
        *_, year, julian_day = filename.split(".")

        file_date = date(int(year), 1, 1) + timedelta(days=int(julian_day) - 1)
        self.file_dates.append(file_date)
        self.channels.add(channel.rstrip(".D"))

    @property
    def start_date(self) -> date:
        return min(self.file_dates)

    @property
    def end_date(self) -> date:
        return max(self.file_dates) + timedelta(days=1)


class SDSArchiveStats(Stats):
    n_files: int = 0
    n_bytes: int = 0


class SDSArchive(WaveformProvider):
    provider: Literal["SDSArchive"] = "SDSArchive"

    archive: DirectoryPath = Field(
        default=Path.cwd() / "sds-archive",
        description="Path to the root of the SDS archive.",
    )

    start_time: AwareDatetime | None = Field(
        default=None,
        description="Start time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-30T00:00:00Z`.",
    )
    end_time: AwareDatetime | None = Field(
        default=None,
        description="End time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-31T00:00:00Z`.",
    )
    n_threads: PositiveInt = Field(
        default=4,
        description="Number of threads to use for reading data from the SDS archive.",
    )

    _archive_stations: dict[NSL, StationCovarage] = PrivateAttr(default_factory=dict)
    _station_selection: Stations | None = PrivateAttr(default=None)
    _executor: ThreadPoolExecutor | None = PrivateAttr(default=None)

    _stats: SDSArchiveStats = PrivateAttr(default_factory=SDSArchiveStats)

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

    async def scan_archive(self) -> None:
        logger.info("scanning SDS archive at %s", self.archive)
        with get_progress() as progress:
            status = progress.add_task(
                f"Scanning SDS archive at [bold]{self.archive}[/bold]"
            )
            for ifile, file in enumerate(self.archive.glob("**/*.[0-9]*")):
                nsl = nsl_from_path(file.name)
                if nsl not in self._archive_stations:
                    self._archive_stations[nsl] = StationCovarage(nsl=nsl)
                station_coverage = self._archive_stations[nsl]
                station_coverage.add_file(file)

                self._stats.n_files += 1
                self._stats.n_bytes += file.stat().st_size

                if ifile % 250:
                    progress.update(status, advance=250)
                    await asyncio.sleep(0.0)

            progress.remove_task(status)

    async def prepare(self, stations: Stations):
        obspy_compat.plant()
        await self.scan_archive()

        archive_start, archive_end = self.available_time_span()
        logger.info("SDS archive time span: %s to %s", archive_start, archive_end)

        self._station_selection = stations
        self._executor = ThreadPoolExecutor(max_workers=self.n_threads)

    def available_time_span(self) -> tuple[datetime, datetime]:
        all_dates = [
            (covarage.start_date, covarage.end_date)
            for covarage in self._archive_stations.values()
        ]
        start_dates, end_dates = zip(*all_dates, strict=False)
        time = datetime.min.time()
        return (
            datetime.combine(min(start_dates), time).astimezone(timezone.utc),
            datetime.combine(min(end_dates), time).astimezone(timezone.utc),
        )

    def available_nsls(self) -> set[NSL]:
        return set(self._archive_stations.keys())

    def _get_available_paths(
        self,
        nsl: NSL,
        date: date,
        remove_duplicate_orientations: bool = True,
    ) -> list[Path]:
        julian_day = date.timetuple().tm_yday
        base_path = self.archive / str(date.year) / nsl.network / nsl.station
        if not base_path.exists():
            return []

        available_files = {}
        for folder in sorted(base_path.iterdir()):
            if not folder.is_dir():
                continue
            channel = folder.name.rstrip(".D")
            if self.channel_selector and channel[:2] not in self.channel_selector:
                continue
            file_name = f"{nsl.pretty}.{channel}.D.{date.year}.{julian_day:03d}"
            file = folder / file_name
            if not file.exists():
                continue
            available_files[channel] = file

        if self.channel_selector:
            sorting = sorted(
                available_files,
                key=lambda cha: self.channel_selector.index(cha[:2]),
            )
            available_files = {cha: available_files[cha] for cha in sorting}

        if remove_duplicate_orientations:
            seen_orientations = set()
            for channel in available_files.copy():
                orientation = channel[-1]
                if orientation in seen_orientations:
                    available_files.pop(channel)
                seen_orientations.add(orientation)

        return list(available_files.values())

    def _get_file_paths(self, dates: Iterable[date]) -> list[Path]:
        paths = []
        for nsl in self._archive_stations:
            for _date in dates:
                available_files = self._get_available_paths(nsl, _date)
                paths.extend(available_files)
        return paths

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> AsyncIterator[WaveformBatch]:
        archive_start, archive_end = self.available_time_span()
        if start_time is None:
            start_time = archive_start
        end_time = self.end_time or archive_end

        i_batch = 0
        while True:
            if start_time >= end_time:
                break
            # if i_batch > 10:
            #     break
            batch_start = start_time
            batch_end = batch_start + window_increment

            trace_start = batch_start - window_padding
            trace_end = batch_end + window_padding
            dates = {d.date() for d in (trace_start, trace_end)}

            paths = self._get_file_paths(dates)
            traces = await _load_files(
                paths,
                start_time=trace_start,
                end_time=trace_end,
                executor=self._executor,
            )

            batch = WaveformBatch(
                start_time=batch_start,
                end_time=batch_end,
                traces=traces,
                i_batch=i_batch,
            )
            batch.clean_traces()
            start_time = batch_end

            if not batch.is_healthy(min_stations=min_stations):
                logger.info(
                    "skipping unhealthy batch %d: %d stations, %d traces",
                    i_batch,
                    batch.n_stations,
                    len(batch.traces),
                )
                i_batch += 1
                continue

            yield batch


if __name__ == "__main__":
    from qseek.models.station import Stations

    logger.setLevel(logging.DEBUG)

    stations = Stations(
        station_xmls=[Path("/home/marius/Development/tmp/qseek/metadata")]
    )

    sds = SDSArchive(archive=Path("/home/marius/Development/tmp/qseek/data"))

    async def run():
        await sds.prepare(stations)

        async for batch in sds.iter_batches(
            window_increment=timedelta(minutes=10),
            window_padding=timedelta(seconds=30),
            min_stations=1,
        ):
            logger.info("%s", batch)

    asyncio.run(run())
