from __future__ import annotations

from nicegui import ui

from qseek.models.detection import EventDetection
from qseek.ui.explorer import RunSource


class Page:
    async def render(self) -> None:
        raise NotImplementedError


class Component:
    name: str = "Component"
    description: str = ""
    icon: str = ""

    async def render(self) -> None:
        with ui.card().classes("w-full flex-wrap col-6 shadow-2"):
            with ui.row().classes("text-h5"):
                if self.icon:
                    ui.icon(self.icon)
                ui.label(self.name)
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()

    def header(self) -> None:
        ui.label(self.name).classes("text-h5")
        ui.html(self.description, tag="div", sanitize=False).classes("text-body2 mb-2")

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
