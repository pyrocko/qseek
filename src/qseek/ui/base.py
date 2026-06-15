from __future__ import annotations

from nicegui import ui
from nicegui.element import Element

from qseek.models.detection import EventDetection
from qseek.ui.explorer import RunSource


class Component(Element):
    title: str = "Component"
    description: str = ""

    def __init__(self):
        super().__init__()
        with ui.card():
            ui.label(self.title).classes("text-h5")
            if self.description:
                ui.html(
                    self.description,
                    tag="div",
                    sanitize=False,
                ).classes("text-body2 mb-2")


class EventComponent(Component):
    title: str = "Event Component"
    description: str = ""

    def __init__(self, event: EventDetection):
        super().__init__()
        self.event = event

    async def plot(self) -> None:
        raise NotImplementedError


class Badge:
    name: str

    def __init__(self, run: RunSource):
        self.run = run

    async def render(self) -> None: ...

    async def view(self) -> None:
        raise NotImplementedError
