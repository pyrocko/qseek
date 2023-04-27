from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.trace import Trace


class IFC(BaseModel):
    def pre_process(self, traces: list[Trace]) -> list[Trace]:
        ...
