"""Background tasks and queues of the waveform pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from pydantic import Field
from pyrocko.io.mseed import save
from pyrocko.trace import Trace

from qseek.images.base import ImageFunction, Picker, WaveformImage
from qseek.models.station import Station, StationInventory
from qseek.pre_processing.frequency_filters import Bandpass
from qseek.pre_processing.module import PreProcessing
from qseek.utils import BackgroundTasks, prefetch
from qseek.waveforms import sds
from qseek.waveforms.sds import SDSArchive
from qseek.waveforms.seedlink.client import SeedLinkClient
from qseek.waveforms.squirrel import PyrockoSquirrel

TIMEOUT = 10.0
DAY_START = datetime(2024, 5, 20, tzinfo=timezone.utc)
WINDOW = timedelta(minutes=5)
PADDING = timedelta(seconds=10)
N_BATCHES = 11


def pending_tasks() -> list[asyncio.Task]:
    """Tasks other than the test itself that have not finished."""
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]


class Source:
    """Async generator of integers that records whether it was closed."""

    def __init__(self, n: int, error: Exception | None = None) -> None:
        self.n = n
        self.error = error
        self.closed = False
        self.n_yielded = 0

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        try:
            for item in range(self.n):
                await asyncio.sleep(0)
                yield item
                self.n_yielded += 1
            if self.error is not None:
                raise self.error
        finally:
            self.closed = True


def prefetched(iterator, maxsize: int = 2):
    return prefetch(iterator, asyncio.Queue(maxsize=maxsize), name="test")


async def collect(iterator) -> list:
    async with asyncio.timeout(TIMEOUT):
        return [item async for item in iterator]


@pytest.mark.asyncio
@pytest.mark.parametrize("maxsize", [1, 2, 100])
async def test_prefetch(maxsize: int) -> None:
    source = Source(20)
    items = await collect(prefetched(aiter(source), maxsize))
    assert items == list(range(20))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("maxsize", [1, 2, 100])
async def test_prefetch_raises(maxsize: int) -> None:
    """The error is raised after the items buffered before it, without hanging."""
    source = Source(5, error=RuntimeError("producer failed"))
    items = []
    with pytest.raises(RuntimeError, match="producer failed"):
        async with asyncio.timeout(TIMEOUT):
            async for item in prefetched(aiter(source), maxsize):
                items.append(item)
    assert items == list(range(5))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_raises_on_full_queue() -> None:
    """The producer fails while the queue is full and the consumer is busy."""
    # The producer fills the queue with 1 and 2 while the consumer holds 0
    source = Source(3, error=RuntimeError("producer failed"))
    iterator = prefetched(aiter(source))
    async with asyncio.timeout(TIMEOUT):
        assert await anext(iterator) == 0
        while not source.closed:
            await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="producer failed"):
            async for _ in iterator:
                pass
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_consumer_stops_early() -> None:
    source = Source(100)
    iterator = prefetched(aiter(source))
    async with asyncio.timeout(TIMEOUT):
        assert await anext(iterator) == 0
        await iterator.aclose()
    # The producer is cancelled and stops at the full queue
    assert source.closed
    assert source.n_yielded <= 4
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_consumer_cancelled() -> None:
    source = Source(100)
    started = asyncio.Event()

    async def consume() -> None:
        iterator = prefetched(aiter(source))
        async with contextlib.aclosing(iterator):
            async for _ in iterator:
                started.set()
                await asyncio.sleep(10.0)

    task = asyncio.create_task(consume())
    async with asyncio.timeout(TIMEOUT):
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_chain_fails() -> None:
    """A failing stage cancels the stages before it and closes the source."""
    source = Source(100)

    async def fail_after(iterator, n: int):
        # Like the stages of the pipeline, the stage closes its source
        async with contextlib.aclosing(iterator):
            async for item in iterator:
                if item == n:
                    raise RuntimeError("stage failed")
                yield item

    first = prefetched(aiter(source))
    second = prefetched(fail_after(first, 3))
    with pytest.raises(RuntimeError, match="stage failed"):
        await collect(prefetched(second))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_logs_error_after_consumer_stopped(caplog) -> None:
    """The producer's error is logged when the consumer stops first."""
    source = Source(2, error=RuntimeError("producer failed"))
    iterator = prefetched(aiter(source), maxsize=4)
    with caplog.at_level(logging.ERROR):
        async with asyncio.timeout(TIMEOUT):
            assert await anext(iterator) == 0
            while not source.closed:
                await asyncio.sleep(0.01)
            await iterator.aclose()
    assert "test failed" in caplog.text
    assert "producer failed" in caplog.text
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_error_not_logged_twice(caplog) -> None:
    source = Source(2, error=RuntimeError("producer failed"))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        await collect(prefetched(aiter(source)))
    assert "test failed" not in caplog.text


@pytest.mark.asyncio
async def test_background_tasks(caplog) -> None:
    done = []

    async def work(name: str, spawn: bool = False) -> None:
        await asyncio.sleep(0.01)
        if spawn:
            tasks.create_task(work("spawned"))
        done.append(name)

    async def fail() -> None:
        raise RuntimeError("task failed")

    with caplog.at_level(logging.ERROR):
        async with asyncio.timeout(TIMEOUT), BackgroundTasks() as tasks:
            tasks.create_task(work("first", spawn=True))
            tasks.create_task(fail(), name="failing")
            done.append("block")

    # Tasks created while exiting are waited for, a failure does not stop the
    # other tasks or the block
    assert sorted(done) == ["block", "first", "spawned"]
    assert "background task failing failed" in caplog.text
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_background_tasks_cancelled_before_start() -> None:
    """A task cancelled before its first step leaves no unawaited coroutine."""

    async def work() -> None:
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="block failed"):
            async with BackgroundTasks() as tasks:
                tasks.create_task(work())
                raise ValueError("block failed")
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.asyncio
async def test_background_tasks_block_raises() -> None:
    """The tasks are cancelled, the error is raised as it is."""
    with pytest.raises(ValueError, match="block failed"):
        async with asyncio.timeout(TIMEOUT), BackgroundTasks() as tasks:
            task = tasks.create_task(asyncio.sleep(10.0))
            await asyncio.sleep(0)
            raise ValueError("block failed")
    assert task.cancelled()
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_background_tasks_cancelled() -> None:
    started = asyncio.Event()
    inner = []

    async def run() -> None:
        async with BackgroundTasks() as tasks:
            inner.append(tasks.create_task(asyncio.sleep(10.0)))
            started.set()
            await asyncio.sleep(10.0)

    task = asyncio.create_task(run())
    async with asyncio.timeout(TIMEOUT):
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert inner[0].cancelled()
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_background_tasks_cancelled_while_waiting() -> None:
    """Cancelled while waiting on exit, the tasks are cancelled too."""
    inner = []

    async def run() -> None:
        async with BackgroundTasks() as tasks:
            inner.append(tasks.create_task(asyncio.sleep(10.0)))

    task = asyncio.create_task(run())
    async with asyncio.timeout(TIMEOUT):
        while not inner:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert inner[0].cancelled()
    assert not pending_tasks()


NESTED_LOOP = """
import asyncio
import nest_asyncio
from qseek.utils import BackgroundTasks, prefetch

nest_asyncio.apply()

async def numbers():
    for i in range(10):
        yield i

async def main():
    async with BackgroundTasks() as tasks:
        task = tasks.create_task(asyncio.sleep(0, result=1))
        queue = asyncio.Queue(maxsize=2)
        items = [i async for i in prefetch(numbers(), queue, name='numbers')]
    assert items == list(range(10)), items
    assert task.result() == 1

asyncio.run(main())
"""


def test_nested_event_loop() -> None:
    """The CLI runs in nested event loops of nest_asyncio."""
    subprocess.run([sys.executable, "-c", NESTED_LOOP], check=True, timeout=60)


class DummyImageFunction(ImageFunction):
    image: Literal["DummyPipeline"] = "DummyPipeline"
    picker: Picker = Field(default_factory=Picker)

    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        await asyncio.sleep(0)
        return [WaveformImage("Dummy", "cake:P", 1.0, traces, 0.1)]

    def get_blinding(self) -> timedelta:
        return timedelta(seconds=0)

    def get_phases(self) -> tuple[str, ...]:
        return ("cake:P",)


@pytest.fixture
def sds_archive(tmp_path: Path) -> Path:
    """One hour of two stations at 10 Hz in an SDS archive."""
    rng = np.random.default_rng(0)
    for station in ("STA", "STB"):
        trace = Trace(
            "XX",
            station,
            "",
            "HHZ",
            tmin=DAY_START.timestamp(),
            deltat=0.1,
            ydata=rng.integers(-500, 500, 36000).astype(np.int32),
        )
        folder = tmp_path / "2024" / "XX" / station / "HHZ.D"
        folder.mkdir(parents=True)
        save([trace], str(folder / f"XX.{station}..HHZ.D.2024.141"))
    return tmp_path


def station_inventory() -> StationInventory:
    return StationInventory(
        stations=[
            Station(network="XX", station=station, lat=0.0, lon=0.0)
            for station in ("STA", "STB")
        ]
    )


async def pipeline(archive: Path, pre_process: bool = True):
    """The pipeline of a search: waveforms, pre-processing and images."""
    provider = SDSArchive(
        archives=[archive],
        end_time=DAY_START + timedelta(hours=1),
        queue_size=2,
    )
    await provider.prepare(station_inventory())
    modules = [Bandpass(bandpass=(0.5, 4.0))] if pre_process else []
    pre_processing = PreProcessing(modules)
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    return DummyImageFunction().iter_images(pre_processing.iter_batches(batches))


@pytest.mark.asyncio
async def test_pipeline(sds_archive: Path) -> None:
    results = await collect(await pipeline(sds_archive))
    assert len(results) == N_BATCHES
    assert all(len(batch.traces) == 2 for _, batch in results)
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_pipeline_without_pre_processing(sds_archive: Path) -> None:
    results = await collect(await pipeline(sds_archive, pre_process=False))
    assert len(results) == N_BATCHES
    assert not pending_tasks()

    # The waveform loading is closed through the image function
    images = await pipeline(sds_archive, pre_process=False)
    async with asyncio.timeout(TIMEOUT):
        await anext(images)
        await images.aclose()
    assert not pending_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["waveforms", "pre-processing", "images"])
async def test_pipeline_stage_fails(sds_archive: Path, monkeypatch, stage: str) -> None:
    """A failure in any stage reaches the consumer, without hanging."""
    calls = 0

    def fail_third(func):
        async def wrapper(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError(f"{stage} failed")
            return await func(*args, **kwargs)

        return wrapper

    if stage == "waveforms":
        monkeypatch.setattr(sds, "_load_files", fail_third(sds._load_files))
    elif stage == "pre-processing":
        monkeypatch.setattr(
            Bandpass, "process_batch", fail_third(Bandpass.process_batch)
        )
    else:
        monkeypatch.setattr(
            DummyImageFunction,
            "process_traces",
            fail_third(DummyImageFunction.process_traces),
        )

    results = []
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        async with asyncio.timeout(TIMEOUT):
            async for item in await pipeline(sds_archive):
                results.append(item)
    assert len(results) == 2
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_pipeline_consumer_fails(sds_archive: Path) -> None:
    """The search fails, the stages are cancelled when the pipeline closes."""
    images = await pipeline(sds_archive)
    with pytest.raises(RuntimeError, match="search failed"):
        async with asyncio.timeout(TIMEOUT), contextlib.aclosing(images):
            async for _ in images:
                raise RuntimeError("search failed")
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_sds_not_prepared(sds_archive: Path) -> None:
    provider = SDSArchive(archives=[sds_archive])
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    with pytest.raises(RuntimeError, match="not prepared"):
        await collect(batches)
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_pre_processing_runs_twice(sds_archive: Path) -> None:
    """Every run has its own queue."""
    for _ in range(2):
        assert len(await collect(await pipeline(sds_archive))) == N_BATCHES


class FakeSquirrel:
    """Chops `n_batches` batches, then raises `error` if given."""

    def __init__(self, n_batches: int, error: Exception | None = None) -> None:
        self.n_batches = n_batches
        self.error = error
        self.chop_tmins: list[float] = []

    def get_time_span(self, kinds: list[str]) -> tuple[float, float]:
        return DAY_START.timestamp(), DAY_START.timestamp() + 3600.0

    def chopper_waveforms(self, tmin: float, tinc: float, **kwargs):
        self.chop_tmins.append(tmin)
        for i in range(self.n_batches):
            start = tmin + i * tinc
            yield SimpleNamespace(
                traces=[
                    Trace(
                        "XX",
                        "STA",
                        "",
                        "HHZ",
                        tmin=start,
                        deltat=1.0,
                        ydata=np.zeros(int(tinc)),
                    )
                ],
                tmin=start,
                tmax=start + tinc,
                i=i,
                n=self.n_batches,
            )
        if self.error is not None:
            raise self.error


def squirrel_provider(squirrel: FakeSquirrel, **kwargs) -> PyrockoSquirrel:
    provider = PyrockoSquirrel(persistent="test", queue_size=2, **kwargs)
    provider._squirrel = squirrel  # ty: ignore[invalid-assignment]
    provider._stations = station_inventory()
    return provider


@pytest.mark.asyncio
async def test_squirrel_batches() -> None:
    provider = squirrel_provider(FakeSquirrel(5))
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    assert [batch.i_batch for batch in await collect(batches)] == list(range(5))
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_squirrel_loading_fails() -> None:
    provider = squirrel_provider(FakeSquirrel(3, error=OSError("bad file")))
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    with pytest.raises(OSError, match="bad file"):
        await collect(batches)
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_squirrel_consumer_stops_early() -> None:
    provider = squirrel_provider(FakeSquirrel(100))
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    async with asyncio.timeout(TIMEOUT):
        await anext(batches)
        await batches.aclose()
    assert not pending_tasks()


class RescanSleeps(list):
    """The sleeps before the rescans, they never end if `block` is set."""

    block = False


@pytest.fixture
def rescan_sleeps(monkeypatch) -> RescanSleeps:
    sleeps = RescanSleeps()
    sleep = asyncio.sleep

    async def fake_sleep(delay: float, result=None):
        if delay == 600.0:
            sleeps.append(delay)
            if sleeps.block:
                await asyncio.Event().wait()
        return await sleep(0, result)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(PyrockoSquirrel, "scan_waveform_dirs", lambda self, sq: None)
    return sleeps


def watching_provider(fake: FakeSquirrel, tmp_path: Path) -> PyrockoSquirrel:
    return squirrel_provider(
        fake, waveform_dirs=[tmp_path], watch_waveforms=timedelta(minutes=10)
    )


@pytest.mark.asyncio
async def test_squirrel_watch_waveforms(tmp_path: Path, rescan_sleeps) -> None:
    """After a pass, the rescan continues at the end of the last batch."""
    fake = FakeSquirrel(3)
    provider = watching_provider(fake, tmp_path)
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    async with asyncio.timeout(TIMEOUT):
        first_pass = [await anext(batches) for _ in range(3)]
        await anext(batches)
        await batches.aclose()

    assert rescan_sleeps == [600.0]
    assert fake.chop_tmins[1] == (first_pass[-1].end_time + PADDING).timestamp()
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_squirrel_watch_cancelled(tmp_path: Path, rescan_sleeps) -> None:
    """The consumer is cancelled while it waits for the rescan."""
    rescan_sleeps.block = True
    provider = watching_provider(FakeSquirrel(3), tmp_path)
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    async with asyncio.timeout(TIMEOUT):
        for _ in range(3):
            await anext(batches)
        task = asyncio.create_task(anext(batches))
        while not rescan_sleeps:
            await asyncio.sleep(0)
        # The loading of the first pass has finished
        assert not [t for t in pending_tasks() if t is not task]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not pending_tasks()


def seedlink_client() -> SeedLinkClient:
    client = SeedLinkClient(host="localhost")
    client._request_selection = set(client.station_selection)
    return client


@pytest.mark.asyncio
async def test_seedlink_check_stream() -> None:
    client = seedlink_client()

    async def fail() -> None:
        raise FileNotFoundError("slinktool")

    client._task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="streaming from localhost") as exc:
        client.check_stream()
    assert isinstance(exc.value.__cause__, FileNotFoundError)

    # Running, finished and cancelled streams pass
    client._task = asyncio.create_task(asyncio.sleep(10.0))
    client.check_stream()
    client._task.cancel()
    await asyncio.sleep(0)
    client.check_stream()
    client._task = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0.01)
    client.check_stream()


@pytest.mark.asyncio
async def test_seedlink_slinktool_exits(monkeypatch) -> None:
    """A slinktool that exits fails the stream instead of spinning."""
    create_subprocess_exec = asyncio.subprocess.create_subprocess_exec

    async def failing_slinktool(*args, **kwargs):
        script = "import sys; sys.stderr.write('bad arguments'); sys.exit(1)"
        proc = await create_subprocess_exec(sys.executable, "-c", script, **kwargs)
        # Exited and reaped before the stream reads, terminate() then raises
        await proc.wait()
        return proc

    monkeypatch.setattr(asyncio.subprocess, "create_subprocess_exec", failing_slinktool)
    client = seedlink_client()
    with pytest.raises(RuntimeError, match="slinktool exited: bad arguments"):
        async with asyncio.timeout(TIMEOUT):
            await client.stream()
