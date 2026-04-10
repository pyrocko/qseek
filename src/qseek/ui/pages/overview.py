from __future__ import annotations

import numpy as np
from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components import statistics as statistics_components
from qseek.ui.components.map import OverviewMap
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


class OverviewPage(Page):
    async def render(self) -> None:
        catalog = await get_tab_state().get_filtered_catalog()

        if catalog.n_events == 0:
            with ui.row().classes("items-center gap-2 text-grey-6 mt-2"):
                ui.icon("info").classes("text-grey-6")
                ui.label("No events found").classes("text-body1 font-medium")
            ui.label(
                "No events are present in this catalog yet. Run the detection pipeline to populate the catalog."
            ).classes("text-grey-6 text-body2")
            return

        n_picks_values = [ev.n_picks for ev in catalog.events if ev.n_picks is not None]

        n_picks_median = np.nanmedian(n_picks_values) if n_picks_values else np.nan
        n_picks_max = np.nanmax(n_picks_values) if n_picks_values else np.nan
        n_picks_min = np.nanmin(n_picks_values) if n_picks_values else np.nan
        semblance_max = np.nanmax(catalog.semblances)
        semblance_min = np.nanmin(catalog.semblances)

        if catalog.has_magnitudes():
            magnitude_max = np.nanmax(catalog.magnitudes)
            magnitude_min = np.nanmin(catalog.magnitudes)
        else:
            magnitude_max = 0.0
            magnitude_min = 0.0

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
                tooltip="Total number of detected events in the catalog.",
            )
            stat_card(
                "Event rate",
                f"{event_rate:.2f}",
                subtitle="events/day",
                icon="timeline",
                tooltip="Average number of detected events per day.",
            )
            stat_card(
                "Median Picks",
                f"{n_picks_median:.0f}",
                icon="scatter_plot",
                subtitle=f"Min: {n_picks_min}, Max: {n_picks_max}",
                tooltip="Median number of picks per event.",
            )
            stat_card(
                "Max Semblance",
                f"{semblance_max:.2f}",
                icon="stacked_line_chart",
                subtitle=f"Min Semblance: {semblance_min:.2f}",
                tooltip="Maximum semblance value among all detected events.",
            )
            if catalog.has_magnitudes():
                stat_card(
                    "Max Magnitude",
                    f"{magnitude_max:.2f}",
                    icon="bar_chart",
                    subtitle=f"Min Magnitude: {magnitude_min:.2f}",
                    tooltip="Maximum magnitude among all detected events.",
                )

        with ui.row().classes("items-center gap-4 w-full"):
            await OverviewMap().render()
            with ui.expansion(
                "Rates",
                icon="insights",
                value=True,
            ).classes("w-full"):
                await statistics_components.SemblanceRate().render()
            with ui.expansion(
                "Migration Plot",
                icon="show_chart",
                value=True,
            ).classes("w-full"):
                await statistics_components.MigrationPlot().render()
            with ui.expansion(
                "Depth Sections",
                icon="vertical_align_center",
                value=True,
            ).classes("w-full"):
                await statistics_components.DepthSection(
                    direction="north-south"
                ).render()
                await statistics_components.DepthSection(direction="east-west").render()
            with ui.expansion(
                "N Picks Distribution",
                icon="bar_chart",
                value=True,
            ).classes("w-full"):
                await statistics_components.NPicksDistribution().render()
