from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, Type
from weakref import WeakValueDictionary

from pydantic import BaseModel, PrivateAttr, create_model
from pydantic.fields import ComputedFieldInfo, FieldInfo
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from lassie.utils import CONSOLE

logger = logging.getLogger(__name__)

STATS_CLASSES: set[Type[Stats]] = set()
STATS_INSTANCES: WeakValueDictionary[str, Stats] = WeakValueDictionary()


PROGRESS = Progress()


def titelify(name: str) -> str:
    return " ".join(word for word in name.split("_")).capitalize()


class RuntimeStats(BaseModel):
    @classmethod
    def new(cls) -> RuntimeStats:
        return create_model(
            "RuntimeStats",
            **{
                stats.__name__: (stats, STATS_INSTANCES.get(stats.__name__, None))
                for stats in STATS_CLASSES
            },
            __base__=cls,
        )()

    def __rich__(self) -> Group:
        return Group(
            *(
                getattr(self, stat_name)
                for stat_name in self.model_fields
                if getattr(self, stat_name, None)
            ),
            PROGRESS,
        )

    async def live_view(self):
        with Live(
            self,
            console=CONSOLE,
            refresh_per_second=10,
            auto_refresh=True,
            redirect_stdout=True,
            redirect_stderr=True,
        ) as _:
            while True:
                await asyncio.sleep(1.0)


class Stats(BaseModel):
    _position: int = PrivateAttr(0)

    def __init_subclass__(cls: Type[Stats], **kwargs) -> None:
        STATS_CLASSES.add(cls)

    def model_post_init(self, __context: Any) -> None:
        STATS_INSTANCES[self.__class__.__name__] = self

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
