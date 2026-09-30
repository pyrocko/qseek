import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from pydantic import Field
from pyrocko.trace import Trace

from qseek.images.base import ImageFunction, Picker, WaveformImage

START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
TIMEOUT = 5.0


class DummyImageFunction(ImageFunction):
    """Returns one P image per batch, or raises the batch's `error`."""

    image: Literal["Dummy"] = "Dummy"
    picker: Picker = Field(default_factory=Picker)

    async def process_traces(self, traces):
        await asyncio.sleep(0)
        error = getattr(traces, "error", None)
        if error is not None:
            raise error
        return [
            WaveformImage("Dummy", "cake:P", 1.0, list(traces), 0.1),
            WaveformImage("Dummy", "cake:S", 1.0, [], 0.1),
        ]

    def get_blinding(self):
        return timedelta(seconds=0)

    def get_phases(self):
        return ("cake:P", "cake:S")


class BatchTraces(list):
    error: Exception | None = None


def batch(n_traces: int = 1, error: Exception | None = None) -> SimpleNamespace:
    traces = BatchTraces(
        Trace("XX", f"S{idx}", "", "Z", 0.0, deltat=0.01, ydata=np.zeros(100))
        for idx in range(n_traces)
    )
    traces.error = error
    return SimpleNamespace(
        is_healthy=lambda: True,
        traces=traces,
        start_time=START_TIME,
        end_time=START_TIME + timedelta(seconds=1),
        nbytes=100,
    )


class Batches:
    """Async batch iterator that records whether it was closed."""

    def __init__(self, batches: list[SimpleNamespace]) -> None:
        self.batches = batches
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        try:
            for item in self.batches:
                yield item
        finally:
            self.closed = True


async def collect(function: ImageFunction, batches: Batches) -> list:
    return [images async for images, _ in function.iter_images(aiter(batches))]


@pytest.mark.asyncio
async def test_iter_images():
    function = DummyImageFunction()
    results = await asyncio.wait_for(
        collect(function, Batches([batch(), batch(3)])), TIMEOUT
    )
    assert [images.n_images for images in results] == [1, 1]
    # The S image without traces is skipped
    assert [images.images[0].phase for images in results] == ["cake:P", "cake:P"]
    assert results[1].images[0].n_traces == 3


@pytest.mark.asyncio
async def test_iter_images_skips_value_error():
    function = DummyImageFunction()
    results = await asyncio.wait_for(
        collect(
            function,
            Batches([batch(error=ValueError("bad batch")), batch(0), batch(2)]),
        ),
        TIMEOUT,
    )
    # The batch raising ValueError and the batch without any traces are skipped
    assert [images.images[0].n_traces for images in results] == [2]


@pytest.mark.asyncio
async def test_iter_images_raises_worker_error():
    function = DummyImageFunction()
    with pytest.raises(RuntimeError, match="processing failed"):
        await asyncio.wait_for(
            collect(
                function,
                Batches([batch(), batch(error=RuntimeError("processing failed"))]),
            ),
            TIMEOUT,
        )


@pytest.mark.asyncio
async def test_iter_images_consumer_stops_early():
    function = DummyImageFunction()
    batches = Batches([batch() for _ in range(100)])
    iterator = function.iter_images(aiter(batches))

    await asyncio.wait_for(anext(iterator), TIMEOUT)
    await asyncio.wait_for(iterator.aclose(), TIMEOUT)

    # The worker is cancelled and the batch iterator is closed
    assert batches.closed
    remaining = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert not remaining
