from nicegui import ui
import numpy as np

from qseek.ui.base import Page
from qseek.ui.components.statistics import (
    SemblanceRate,
    MagnitudeRate,
    MigrationPlot,
    DepthSection,
)
from qseek.ui.utils import stat_card
from qseek.ui.components.map import OverviewMap
from qseek.ui.components.magnitudes import MagnitudeFrequency, MagnitudeSemblance


class OverviewPage(Page):
    async def render(self) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()

        median_n_picks = np.nanmedian([ev.n_picks for ev in catalog.events])
        max_n_picks = np.nanmax([ev.n_picks for ev in catalog.events])
        min_n_picks = np.nanmin([ev.n_picks for ev in catalog.events])
        semblance_median = np.nanmedian([ev.semblance for ev in catalog.events])
        semblance_max = np.nanmax([ev.semblance for ev in catalog.events])
        min_semblance = np.nanmin([ev.semblance for ev in catalog.events])
        median_magnitude = np.nanmedian([ev.magnitude for ev in catalog.events])
        min_magnitude = np.nanmin([ev.magnitude for ev in catalog.events])
        max_magnitude = np.nanmax([ev.magnitude for ev in catalog.events])
        median_depth = np.nanmedian([ev.depth for ev in catalog.events])
        max_depth = np.nanmax([ev.depth for ev in catalog.events])
        min_depth = np.nanmin([ev.depth for ev in catalog.events])
        event_rate = len(catalog.events) / (
            (catalog.times[-1] - catalog.times[0]).total_seconds() / (3600 * 24)
        )
        num_days = int(
            (catalog.times[-1] - catalog.times[0]).total_seconds() / (3600 * 24)
        )
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label("Overview").classes("text-h1")
            ui.space()
            ui.chip(
                str(
                    f"Time range: {catalog.times[0].date()} - {catalog.times[-1].date()}"
                ),
                icon="schedule",
            ).props("outline").classes("text-xs font-mono text-grey-9")

        with ui.row().classes("items-center gap-4 w-full"):
            stat_card(
                "Total Events",
                str(catalog.n_events),
                icon="crisis_alert",
                subtitle=f"over {num_days:.0f} days",
            )
            stat_card(
                "Event rate",
                f"{event_rate:.2f} events/day",
                icon="timeline",
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
                subtitle=f"Min Semblance: {min_semblance:.2f}",
            )
            stat_card(
                "Max Magnitude",
                f"{max_magnitude:.2f}",
                icon="bar_chart",
                subtitle=f"Min Magnitude: {min_magnitude:.2f}",
            )

        with ui.row().classes("items-center gap-4 w-full"):
            await SemblanceRate(run).render()
            await OverviewMap(run).render()
            await MigrationPlot(run).render()
            await DepthSection(run, direction="north-south").render()
            await DepthSection(run, direction="east-west").render()

        if True:
            ui.label("Magnitudes").classes("text-h2")
            with ui.row().classes("items-center gap-4 w-full"):
                await MagnitudeFrequency(run).render()
                await MagnitudeSemblance(run).render()
                await MagnitudeRate(run).render()
