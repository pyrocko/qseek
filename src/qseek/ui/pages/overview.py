from nicegui import ui
import numpy as np

from qseek.ui.base import Page
from qseek.ui.components.statistics import SemblanceRate, MagnitudeRate
from qseek.ui.utils import stat_card
from qseek.ui.components.map import OverviewMap
from qseek.ui.components.magnitudes import MagnitudeFrequency, MagnitudeSemblance


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()

        median_n_picks = np.nanmedian([ev.n_picks for ev in catalog.events])
        semblance_median = np.nanmedian([ev.semblance for ev in catalog.events])
        semblance_max = np.nanmax([ev.semblance for ev in catalog.events])
        median_magnitude = np.nanmedian([ev.magnitude for ev in catalog.events])
        median_depth = np.nanmedian([ev.depth for ev in catalog.events])

        ui.label("Overview").classes("text-h1")
        with ui.row().classes("items-center gap-4 w-full"):
            stat_card(
                "Total Events",
                str(catalog.n_events),
                icon="crisis_alert",
            )
            stat_card(
                "Median Picks",
                f"{median_n_picks:.1f}",
                icon="scatter_plot",
            )
            stat_card(
                "Max Semblance",
                f"{semblance_max:.2f}",
                icon="stacked_line_chart",
            )
            stat_card(
                "Median Magnitude",
                f"{median_magnitude:.2f}",
                icon="bar_chart",
            )
            stat_card(
                "Median Depth",
                f"{median_depth:.1f} m",
                icon="vertical_align_bottom",
            )

        with ui.row().classes("items-center gap-4 w-full"):
            await SemblanceRate(run).render()
            await OverviewMap(run).render()

        if True:
            ui.label("Magnitudes").classes("text-h2")
            with ui.row().classes("items-center gap-4 w-full"):
                await MagnitudeFrequency(run).render()
                await MagnitudeSemblance(run).render()
                await MagnitudeRate(run).render()
