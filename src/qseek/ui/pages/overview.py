from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.statistics import SemblanceRate


def stat_card(
    label: str, value: str, icon: str = "info", color: str = "#00b4d8"
) -> None:
    with (
        ui.card()
        .classes("p-4 items-center text-center")
        .style(f"background:#16213e; border:1px solid {color}44; min-width:140px;")
    ):
        ui.icon(icon).style(f"color:{color}; font-size:2rem;")
        ui.label(value).classes("text-h6 text-weight-bold").style(f"color:{color}")
        ui.label(label).classes("text-caption text-grey-5")


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()

        ui.label("Overview").classes("text-h4 mb-4")
        stat_card(
            "Total Events", str(catalog.n_events), icon="crisis_alert", color="#00b4d8"
        )

        with ui.row().classes("w-full"):
            await SemblanceRate(run).render()
            await SemblanceRate(run).render()
