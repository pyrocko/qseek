from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

from qseek.models.detection import EventDetection
from qseek.ui.explorer import RunSource

if TYPE_CHECKING:
    from qseek.ui.state import CatalogStore


class Component:
    name: str = "Component"
    description: str = ""
    icon: str = ""

    def __init__(self, catalog_store: CatalogStore):
        self.catalog = catalog_store

    def header(self, title: str = "", description: str = "") -> None:
        ui.label(title or self.name).classes("text-h5")
        ui.html(
            description or self.description,
            tag="div",
            sanitize=False,
        ).classes("text-body2 mb-2")

    async def view(self) -> None:
        raise NotImplementedError


class EventComponent(Component):
    name: str = "Event Component"
    description: str = ""

    def __init__(self, event: EventDetection):
        self.event = event

    async def view(self) -> None:
        raise NotImplementedError


class Badge:
    name: str

    def __init__(self, run: RunSource):
        self.run = run

    async def render(self) -> None: ...

    async def view(self) -> None:
        raise NotImplementedError
