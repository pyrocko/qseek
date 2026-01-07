from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Self

from pydantic import (
    DirectoryPath,
    Field,
    PositiveInt,
    PrivateAttr,
    computed_field,
    model_validator,
)
from pyrocko import obspy_compat
from pyrocko.io.mseed import iload
from pyrocko.trace import degapper

from qseek.stats import Stats, get_progress
from qseek.utils import (
    NSL,
    QUEUE_SIZE,
    DateTime,
    datetime_now,
    human_readable_bytes,
    setup_rich_logging,
)
from qseek.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table

    from qseek.models.station import StationInventory


logger = logging.getLogger(__name__)
ANY = "[A-Z0-9]"
NETWORK = f"{ANY}{ANY}"
STATION = f"{ANY}*"
CHANNEL = f"{ANY}{ANY}{ANY}.D"
JDAY = "[0-9]*"


async def _load_file(
    file: Path,
    start_time: datetime,
    end_time: datetime,
    executor: ThreadPoolExecutor | None = None,
) -> list[Trace]:
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()

    def load_traces() -> list[Trace]:
        # logger.debug("loading file %s", file)
        return list(
            iload(
                str(file),
                tmin=start_time.timestamp(),
                tmax=end_time.timestamp(),
            )
        )

    func_call = partial(
        ctx.run,
        load_traces,
    )
    return await loop.run_in_executor(executor, func_call)


async def _load_files(
    files: Iterable[Path],
    start_time: datetime,
    end_time: datetime,
    want_incomplete: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> list[Trace]:
    try:
        result = await asyncio.gather(
            *[
                _load_file(file, start_time, end_time, executor=executor)
                for file in files
            ],
            return_exceptions=True,
        )
    except Exception as e:
        logger.error("error loading files: %s", e)
        return []

    for exc in (tr for tr in result if isinstance(tr, Exception)):
        logger.error("error loading file: %s", exc)

    traces = [tr for tr in result if not isinstance(tr, Exception)]
    traces = list(chain(*traces))

    if not traces:
        logger.warning("no traces loaded from files")

    degapped_traces = degapper(sorted(traces, key=lambda tr: tr.full_id))

    if not want_incomplete:
        for tr in degapped_traces.copy():
            start_offset = abs(tr.tmin - start_time.timestamp())
            end_offset = abs(tr.tmax - end_time.timestamp() + tr.deltat)
            if not (start_offset <= 0.5 * tr.deltat and end_offset <= 0.5 * tr.deltat):
                degapped_traces.remove(tr)

    return degapped_traces


def _nsl_from_filename(name: str) -> NSL:
    network, station, location, *_ = name.split(".")
    return NSL(network, station, location)


class StationCovarage(NamedTuple):
    nsl: NSL
    channels: set[str] = set()
    file_dates: list[date] = []

    def add_file(self, file: Path):
        *_, channel, filename = file.parts
        # if self.nsl != _nsl_from_filename(filename):
        #     raise ValueError(f"File {file} does not match station {self.nsl.pretty}")
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
    n_files_scanned: int = 0
    n_bytes_scanned: int = 0
    time_per_batch: timedelta = timedelta(seconds=0.0)
    bytes_per_seconds: float = 0.0

    _queue: asyncio.Queue[WaveformBatch | None] | None = PrivateAttr(None)
    _position = 20
    _show_header = False

    def set_queue(self, queue: asyncio.Queue[WaveformBatch | None]) -> None:
        self._queue = queue

    @computed_field
    @property
    def queue_size(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @computed_field
    @property
    def queue_size_max(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.maxsize

    def _populate_table(self, table: Table) -> None:
        alert = self.queue_size <= 2
        prefix, suffix = ("[bold red]", "[/bold red]") if alert else ("", "")
        table.add_row(
            "[bold]Waveform loading[/bold]",
            f"Q:{prefix}{self.queue_size:>2}/{self.queue_size_max}{suffix}"
            f" {human_readable_bytes(self.bytes_per_seconds) + '/s':>10}",
        )


class SDSArchive(WaveformProvider):
    """SDSArchive waveform provider for reading data from a local SDS archive."""

    provider: Literal["SDSArchive"] = "SDSArchive"

    archive: DirectoryPath = Field(
        default=Path.cwd() / "sds-archive",
        description="Path to the root of the SDS archive.",
    )

    start_time: DateTime | None = Field(
        default=None,
        description="Start time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-30T00:00:00Z`.",
    )
    end_time: DateTime | None = Field(
        default=None,
        description="End time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-31T00:00:00Z`.",
    )

    n_threads: PositiveInt = Field(
        default=4,
        description="Number of threads to use for reading data from the SDS archive.",
    )
    queue_size: PositiveInt = Field(
        default=QUEUE_SIZE,
        description="Maximum number of waveform batches to keep in the internal queue.",
    )

    _archive_stations: dict[NSL, StationCovarage] = PrivateAttr(default_factory=dict)
    _station_selection: StationInventory | None = PrivateAttr(default=None)

    _queue: asyncio.Queue[WaveformBatch | None] = PrivateAttr(
        default_factory=lambda: asyncio.Queue(maxsize=QUEUE_SIZE)
    )
    _executor: ThreadPoolExecutor | None = PrivateAttr(default=None)

    _stats: SDSArchiveStats = PrivateAttr(default_factory=SDSArchiveStats)

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

    def scan_sds_archive(self) -> None:
        logger.info("scanning SDS archive at %s", self.archive)

        if self.start_time:
            end_year = self.end_time.year if self.end_time else datetime_now().year
            years = range(self.start_time.year, end_year + 1)
            sds_iter = chain(
                *(
                    self.archive.glob(f"{year}/{NETWORK}/{STATION}/{CHANNEL}/*.{JDAY}")
                    for year in years
                )
            )
        else:
            sds_iter = self.archive.glob(f"**/{NETWORK}/{STATION}/{CHANNEL}/*.{JDAY}")

        n_files = 0
        start = datetime_now()
        with get_progress() as progress:
            status = progress.add_task(
                f"Scanning SDS archive at [bold]{self.archive}[/bold]",
                total=None,
            )
            for file in sds_iter:
                nsl = _nsl_from_filename(file.name)
                if nsl not in self._archive_stations:
                    self._archive_stations[nsl] = StationCovarage(nsl=nsl)
                station_coverage = self._archive_stations[nsl]
                station_coverage.add_file(file)

                self._stats.n_files_scanned += 1
                self._stats.n_bytes_scanned += file.stat().st_size
                n_files += 1

                if n_files % 500 == 0:
                    progress.update(status, advance=500)

            progress.remove_task(status)

        self._archive_stations = {
            nsl: self._archive_stations[nsl] for nsl in sorted(self._archive_stations)
        }

        if n_files == 0:
            raise EnvironmentError(f"No files found in SDS archive at {self.archive}")
        logger.info(
            "scanned SDS archive in %s, found %s in %d files",
            datetime_now() - start,
            human_readable_bytes(self._stats.n_bytes_scanned),
            n_files,
        )

    def prepare(self, stations: StationInventory):
        obspy_compat.plant()

        self.scan_sds_archive()

        archive_start, archive_end = self.available_time_span()
        logger.info(
            "SDS archive time span: %s to %s",
            archive_start.date(),
            archive_end.date(),
        )

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

    def _get_available_files(
        self,
        nsl: NSL,
        date: date,
        remove_duplicate_orientations: bool = True,
    ) -> set[Path]:
        julian_day = date.timetuple().tm_yday
        base_path = self.archive / str(date.year) / nsl.network / nsl.station
        if not base_path.exists():
            return set()

        available_files: dict[str, Path] = {}
        for folder in sorted(base_path.iterdir()):
            if not folder.is_dir():
                continue
            channel = folder.name.rstrip(".D")
            channel_type = channel[:2]
            if self.channel_selector and channel_type not in self.channel_selector:
                continue
            file_name = f"{nsl.pretty}.{channel}.D.{date.year}.{julian_day:03d}"
            file = folder / file_name
            if not file.exists():
                # This is a fallback for non-zero padded julian days
                file_name = f"{nsl.pretty}.{channel}.D.{date.year}.{julian_day}"
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
                cha_orientation = channel[-1]
                if cha_orientation in seen_orientations:
                    available_files.pop(channel)
                seen_orientations.add(cha_orientation)

        return set(available_files.values())

    def _get_file_paths(self, dates: Iterable[date]) -> set[Path]:
        if not self._station_selection:
            raise RuntimeError("Station selection not prepared.")

        paths: set[Path] = set()
        for sta in self._station_selection:
            for _date in dates:
                files = self._get_available_files(sta.nsl, _date)
                paths.update(files)

        logger.debug(
            "found %d MiniSeed files in SDS archive for %s",
            len(paths),
            ", ".join(str(d) for d in dates),
        )
        return paths

    async def _fetch_waveform_data(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> None:
        logger.info("starting SDS waveform reader, queue size %d", self.queue_size)
        stats = self._stats

        archive_start, archive_end = self.available_time_span()
        start_time = start_time or archive_start + window_padding
        end_time = end_time or archive_end

        n_batches = int((end_time - start_time) // window_increment)
        i_batch = 0
        while True:
            if start_time >= end_time:
                break

            batch_start = start_time
            batch_end = min(batch_start + window_increment, end_time)

            trace_start = batch_start - window_padding
            trace_end = batch_end + window_padding

            dates = {trace_start.date(), trace_end.date()}

            begin_load = datetime_now()
            paths = self._get_file_paths(dates)
            if not paths:
                logger.warning("no waveform files found for %s, skipping", trace_start)
                i_batch += 1
                start_time = batch_end
                await asyncio.sleep(0.0)
                continue

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
                n_batches=n_batches,
            )
            stats.time_per_batch = datetime_now() - begin_load
            stats.bytes_per_seconds = (
                batch.nbytes / stats.time_per_batch.total_seconds()
            )

            batch.clean_traces()
            i_batch += 1
            start_time = batch_end

            if not batch.is_healthy(min_stations=min_stations):
                logger.info(
                    "skipping unhealthy batch %s: %d stations, %d traces",
                    start_time,
                    batch.n_stations,
                    len(batch.traces),
                )
                continue

            if min_length and (trace_end - trace_start) < min_length:
                logger.info(
                    "skipping short batch %s: %.1f s, required %.1f s",
                    start_time,
                    batch.duration.total_seconds(),
                    min_length.total_seconds(),
                )
                continue

            logger.debug(
                "loaded waveforms for batch %d/%d at %s",
                i_batch,
                n_batches,
                human_readable_bytes(stats.bytes_per_seconds) + "/s",
            )
            await self._queue.put(batch)

        await self._queue.put(None)
        logger.info("waveform data fetcher task completed")

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> AsyncGenerator[WaveformBatch]:
        self._queue = asyncio.Queue(maxsize=self.queue_size)
        self._stats.set_queue(self._queue)

        worker = asyncio.create_task(
            self._fetch_waveform_data(
                window_increment=window_increment,
                window_padding=window_padding,
                start_time=start_time or self.start_time,
                end_time=self.end_time,
                min_length=min_length,
                min_stations=min_stations,
            )
        )

        while True:
            batch = await self._queue.get()
            if batch is None:
                self._queue.task_done()
                break
            yield batch
            self._queue.task_done()

        await worker


if __name__ == "__main__":
    from cProfile import Profile

    from qseek.models.station import StationInventory

    p = Profile()

    setup_rich_logging(logging.DEBUG)
    stations = StationInventory(
        station_xmls=[Path("/project/elise-info/sds/ELISE.xml")]
    )

    sds = SDSArchive(
        archive=Path("/project/elise-info/sds/"),
        n_threads=16,
        start_time=datetime(2025, 8, 20, tzinfo=timezone.utc),
    )

    async def run():
        sds.prepare(stations)
        # return
        p.enable()
        async for batch in sds.iter_batches(
            window_increment=timedelta(minutes=10),
            window_padding=timedelta(seconds=30),
            min_stations=1,
        ):
            batch.clean_traces()
        p.disable()
        p.dump_stats("/tmp/sds.prof")

    asyncio.run(run())
