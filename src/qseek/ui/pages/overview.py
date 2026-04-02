from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.statistics import SemblanceRate, MagnitudeRate
from qseek.ui.utils import stat_card
from qseek.ui.components.map import OverviewMap


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()

        ui.label("Overview").classes("text-h4 mb-4")
        with ui.row().classes("items-center gap-4 w-full"):
            stat_card(
                "Total Events",
                str(catalog.n_events),
                icon="crisis_alert",
            )
            stat_card(
                "Median Picks",
                f"{catalog.median_n_picks:.1f}",
                icon="scatter_plot",
            )
            stat_card(
                "Median Semblance",
                f"{catalog.median_semblance:.2f}",
                icon="stacked_line_chart",
            )
            stat_card(
                "Median Magnitude",
                f"{catalog.median_magnitude:.2f}",
                icon="bar_chart",
            )
            stat_card(
                "Median Depth",
                f"{catalog.median_depth:.1f} m",
                icon="vertical_align_bottom",
            )

        with ui.row().classes("items-center gap-2 w-full"):
            await SemblanceRate(run).render()
            await MagnitudeRate(run).render()
            await OverviewMap(run).render()
