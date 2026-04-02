from nicegui import ui

from qseek.ui.models import RunManager, RunProxy


class Page:
    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager

    async def render(self) -> None:
        raise NotImplementedError


class Component:
    name: str = "Component"
    description: str = ""

    def __init__(self, run: RunProxy):
        self.run = run

    async def render(self) -> None:
        with ui.card().classes("flex-1 min-w-md h-full col-6 shadow-2"):
            ui.label(self.name).classes("text-h5")
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()

    async def view(self) -> None:
        raise NotImplementedError


class Badge:
    name: str

    def __init__(self, run: RunProxy):
        self.run = run

    async def render(self) -> None: ...

    async def view(self) -> None:
        raise NotImplementedError
