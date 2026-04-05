from nicegui import ui

from qseek.models.detection import EventDetection
from qseek.ui.explorer import RunSource
from qseek.ui.manager import SourceManager


class Page:
    def __init__(self, run_manager: SourceManager):
        self.run_manager = run_manager

    async def render(self) -> None:
        raise NotImplementedError


class Component:
    name: str = "Component"
    description: str = ""

    def __init__(self, run: RunSource):
        self.run = run

    async def render(self) -> None:
        with ui.card().classes("flex-1 min-w-md col-6 shadow-2"):
            ui.label(self.name).classes("text-h5")
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()

    def header(self) -> None:
        ui.label(self.name).classes("text-h5")
        ui.label(self.description).classes("text-body2 mb-2")

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
