from __future__ import annotations

import asyncio
import logging
from typing import Iterator, Type

from pydantic import BaseModel, create_model
from pydantic.fields import ComputedFieldInfo, FieldInfo
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

logger = logging.getLogger(__name__)

STATS_CLASSES: set[Type[Stats]] = set()


PROGRESS = Progress()


def titelify(name: str) -> str:
    return " ".join(word for word in name.split("_")).capitalize()


class RuntimeStats(BaseModel):
    @classmethod
    def new(cls) -> RuntimeStats:
        return create_model(
            "RuntimeStats",
            **{stats.__name__: (stats, None) for stats in STATS_CLASSES},
            __base__=cls,
        )()

    def __rich__(self) -> Group:
        return Group(
            *(getattr(self, stat_name) for stat_name in self.model_fields_set),
            PROGRESS,
        )

    def add_stats(self, stats: Stats) -> None:
        logger.debug("Adding stats %s", stats.__class__.__name__)
        if stats.__class__.__name__ not in self.model_fields:
            raise ValueError(f"{stats.__class__.__name__} is not a valid stats name")
        if stats.__class__.__name__ in self.model_fields_set:
            raise ValueError(f"{stats.__class__.__name__} is already set")
        setattr(self, stats.__class__.__name__, stats)

    async def live_view(self):
        with Live(
            self,
            refresh_per_second=10,
            screen=True,
            auto_refresh=True,
            redirect_stdout=True,
            redirect_stderr=True,
        ) as _:
            while True:
                await asyncio.sleep(1.0)


class Stats(BaseModel):
    def __init_subclass__(cls: Type[Stats], **kwargs) -> None:
        STATS_CLASSES.add(cls)

    def populate_table(self, table: Table) -> None:
        for name, field in self.iter_fields():
            title = field.title or titelify(name)
            table.add_row(title, str(getattr(self, name)))

    def iter_fields(self) -> Iterator[tuple[str, FieldInfo | ComputedFieldInfo]]:
        yield from self.model_fields.items()
        yield from self.model_computed_fields.items()

    def __rich__(self) -> Panel:
        table = Table(box=None, row_styles=["", "dim"])
        self.populate_table(table)
        return Panel(table, title=self.__class__.__name__)
