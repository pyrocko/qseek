from __future__ import annotations

import asyncio
import logging
import random
import string
from contextlib import contextmanager
from typing import Any, Generator, Iterator
from weakref import WeakValueDictionary

from pydantic import BaseModel, PrivateAttr, computed_field
from pydantic.fields import ComputedFieldInfo, FieldInfo
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from qseek.utils import human_readable_bytes

logger = logging.getLogger(__name__)

STATS_INSTANCES: WeakValueDictionary[str, Stats] = WeakValueDictionary()

_PROGRESS = None
PROGRESS = Progress()


def disable_progress(disable: bool = True) -> None:
    with get_progress() as progress:
        progress.disable = disable


@contextmanager
def get_progress() -> Generator[Progress, None, None]:
    if _PROGRESS is None:
        with Progress() as progress:
            yield progress
    else:
        try:
            yield _PROGRESS
        finally:
            pass


def titelify(name: str) -> str:
    return " ".join(word for word in name.split("_")).capitalize()


class RuntimeStats(BaseModel):
    @classmethod
    async def live_view(cls) -> None:
        global _PROGRESS
        _PROGRESS = Progress()

        def generate_grid() -> Table:
            """Make a new table."""
            table = Table(show_header=False, box=None, expand=True)
            stats_instaces = sorted(
                STATS_INSTANCES.values(),
                key=lambda s: s._position,
            )
            for stats in stats_instaces:
                if stats._show_header:
                    table.add_row(
                        f"{stats.__class__.__name__.removesuffix('Stats')}",
                        style="bold",
                    )
                table.add_section()
                stats._populate_table(table)

            grid = Table.grid(expand=True)
            grid.add_row(PROGRESS)
            grid.add_row(Panel(table, title="QSeek"))
            return grid

        with Live(
            refresh_per_second=4,
            # screen=True,
        ) as live:
            while True:
                live.update(generate_grid())
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    break


class Stats(BaseModel):
    _position: int = PrivateAttr(1)
    _show_header: bool = PrivateAttr(True)

    @classmethod
    def get_subclasses(cls) -> set[type[Stats]]:
        return set(cls.__subclasses__())

    def model_post_init(self, __context: Any) -> None:
        uid = "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        STATS_INSTANCES[uid] = self

    def _populate_table(self, table: Table) -> None:
        for name, field in self.iter_fields():
            title = field.title or titelify(name)
            table.add_row(
                title,
                str(getattr(self, name)),
                style="dim",
            )

    def iter_fields(self) -> Iterator[tuple[str, FieldInfo | ComputedFieldInfo]]:
        yield from self.model_fields.items()
        yield from self.model_computed_fields.items()

    def __rich__(self) -> Panel:
        table = Table(box=None, row_styles=["", "dim"])
        self._populate_table(table)
        return Panel(table, title=self.__class__.__name__)


class QueueStats(Stats):
    """Statistics of a pipeline stage that buffers its output in a queue."""

    _queue: asyncio.Queue | None = PrivateAttr(None)

    def set_queue(self, queue: asyncio.Queue | None) -> None:
        self._queue = queue

    @computed_field
    @property
    def queue_size(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @computed_field
    @property
    def queue_size_max(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.maxsize

    def _add_queue_row(self, table: Table, title: str, bytes_per_second: float) -> None:
        alert = self.queue_size <= 2
        prefix, suffix = ("[bold red]", "[/bold red]") if alert else ("", "")
        table.add_row(
            f"[bold]{title}[/bold]",
            f"Q:{prefix}{self.queue_size:>2}/{self.queue_size_max}{suffix}"
            f" {human_readable_bytes(bytes_per_second) + '/s':>10}",
        )
