from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.statistics import SemblanceRate
from qseek.ui.utils import stat_card


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()

        ui.label("Overview").classes("text-h4 mb-4")
        stat_card(
            "Total Events",
            str(catalog.n_events),
            icon="crisis_alert",
        )

        with ui.row().classes("items-center gap-2 w-full"):
            await SemblanceRate(run).render()
            await SemblanceRate(run).render()
