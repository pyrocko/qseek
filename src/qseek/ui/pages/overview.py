from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.statistics import SemblanceRate


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()

        ui.label("Overview").classes("text-h4 mb-4")
        with ui.row().classes("w-full"):
            await SemblanceRate(run).render()
            await SemblanceRate(run).render()
