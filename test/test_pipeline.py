"""Background tasks and queues of the waveform pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
from qseek.utils import BackgroundTasks, aclose_iterator, aclosing, prefetch
from qseek.waveforms import sds
from qseek.waveforms.sds import SDSArchive
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


async def collect(iterator) -> list:
    async with asyncio.timeout(TIMEOUT):
        return [item async for item in iterator]


@pytest.mark.asyncio
@pytest.mark.parametrize("maxsize", [1, 2, 100])
async def test_prefetch(maxsize: int) -> None:
    source = Source(20)
    items = await collect(prefetch(aiter(source), asyncio.Queue(maxsize=maxsize)))
    assert items == list(range(20))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_plain_iterator() -> None:
    class Counter:
        def __init__(self) -> None:
            self.i = 0

        def __aiter__(self):
            return self

        async def __anext__(self) -> int:
            if self.i == 3:
                raise StopAsyncIteration
            self.i += 1
            return self.i

    assert await collect(prefetch(Counter())) == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("maxsize", [1, 2, 100])
async def test_prefetch_raises(maxsize: int) -> None:
    """The error is raised after the items buffered before it, without hanging."""
    source = Source(5, error=RuntimeError("producer failed"))
    items = []
    with pytest.raises(RuntimeError, match="producer failed"):
        async with asyncio.timeout(TIMEOUT):
            async for item in prefetch(aiter(source), asyncio.Queue(maxsize=maxsize)):
                items.append(item)
    assert items == list(range(5))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_prefetch_raises_on_full_queue() -> None:
    """The producer fails while the queue is full and the consumer is busy."""
    # The producer fills the queue with 1 and 2 while the consumer holds 0
    source = Source(3, error=RuntimeError("producer failed"))
    iterator = prefetch(aiter(source), asyncio.Queue(maxsize=2))
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
    iterator = prefetch(aiter(source), asyncio.Queue(maxsize=2))
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
        iterator = prefetch(aiter(source), asyncio.Queue(maxsize=2))
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

    first = prefetch(aiter(source), asyncio.Queue(maxsize=2))
    second = prefetch(fail_after(first, 3), asyncio.Queue(maxsize=2))
    with pytest.raises(RuntimeError, match="stage failed"):
        await collect(prefetch(second))
    assert source.closed
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_background_tasks_wait_all(caplog) -> None:
    done = []

    async def work(name: str, spawn: bool = False) -> None:
        await asyncio.sleep(0.01)
        if spawn:
            BackgroundTasks.create_task(work("spawned"))
        done.append(name)

    async def fail() -> None:
        raise RuntimeError("task failed")

    BackgroundTasks.create_task(work("first", spawn=True))
    BackgroundTasks.create_task(fail(), name="failing")
    with caplog.at_level(logging.ERROR):
        async with asyncio.timeout(TIMEOUT):
            await BackgroundTasks.wait_all()
        # Done callbacks run on the next loop iteration
        await asyncio.sleep(0)

    # Tasks created while waiting are waited for, a failure does not stop them
    assert sorted(done) == ["first", "spawned"]
    assert "background task failing failed" in caplog.text
    assert not BackgroundTasks.tasks


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


async def pipeline(archive: Path):
    """The pipeline of a search: waveforms, pre-processing and images."""
    provider = SDSArchive(
        archives=[archive],
        end_time=DAY_START + timedelta(hours=1),
        queue_size=2,
    )
    await provider.prepare(station_inventory())
    pre_processing = PreProcessing([Bandpass(bandpass=(0.5, 4.0))])
    batches = provider.iter_batches(window_increment=WINDOW, window_padding=PADDING)
    return DummyImageFunction().iter_images(pre_processing.iter_batches(batches))


@pytest.mark.asyncio
async def test_pipeline(sds_archive: Path) -> None:
    results = await collect(await pipeline(sds_archive))
    assert len(results) == N_BATCHES
    assert all(len(batch.traces) == 2 for _, batch in results)
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
        async with asyncio.timeout(TIMEOUT), aclosing(images):
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

    def get_time_span(self, kinds: list[str]) -> tuple[float, float]:
        return DAY_START.timestamp(), DAY_START.timestamp() + 3600.0

    def chopper_waveforms(self, tmin: float, tinc: float, **kwargs):
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


def squirrel_provider(squirrel: FakeSquirrel) -> PyrockoSquirrel:
    provider = PyrockoSquirrel(persistent="test", queue_size=2)
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
        await aclose_iterator(batches)
    assert not pending_tasks()
