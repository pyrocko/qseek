import numpy as np
from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.magnitudes import (
    MagnitudeFrequency,
    MagnitudeSemblance,
    MagnitudeRate,
)
from qseek.ui.components.map import OverviewMap
from qseek.ui.components.statistics import (
    DepthSection,
    MigrationPlot,
    SemblanceRate,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


class OverviewPage(Page):
    async def render(self) -> None:
        run = get_tab_state().run
        catalog = await run.get_catalog()
        n_picks_values = [ev.n_picks for ev in catalog.events if ev.n_picks is not None]
        semblance_values = [
            ev.semblance for ev in catalog.events if ev.semblance is not None
        ]
        magnitude_values = [
            ev.magnitude.average
            for ev in catalog.events
            if ev.magnitude is not None and ev.magnitude.average is not None
        ]
        # print(catalog.events[0].magnitude)  # Debug print
        has_magnitude = len(magnitude_values) > 0
        median_n_picks = np.nanmedian(n_picks_values) if n_picks_values else np.nan
        max_n_picks = np.nanmax(n_picks_values) if n_picks_values else np.nan
        min_n_picks = np.nanmin(n_picks_values) if n_picks_values else np.nan
        semblance_max = np.nanmax(semblance_values) if semblance_values else np.nan
        min_semblance = np.nanmin(semblance_values) if semblance_values else np.nan
        if has_magnitude:
            min_magnitude = np.nanmin(magnitude_values)
            max_magnitude = np.nanmax(magnitude_values)

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
                f"{median_n_picks:.0f}",
                icon="scatter_plot",
                subtitle=f"Min: {min_n_picks}, Max: {max_n_picks}",
                tooltip="Median number of picks per event.",
            )
            stat_card(
                "Max Semblance",
                f"{semblance_max:.2f}",
                icon="stacked_line_chart",
                subtitle=f"Min Semblance: {min_semblance:.2f}",
                tooltip="Maximum semblance value among all detected events.",
            )
            if has_magnitude:
                stat_card(
                    "Max Magnitude",
                    f"{max_magnitude:.2f}",
                    icon="bar_chart",
                    subtitle=f"Min Magnitude: {min_magnitude:.2f}",
                    tooltip="Maximum magnitude among all detected events.",
                )

        with ui.row().classes("items-center gap-4 w-full"):
            await OverviewMap().render()
            with ui.expansion(
                "Rates",
                icon="insights",
                value=True,
            ).classes("w-full"):
                await SemblanceRate().render()
            with ui.expansion(
                "Migration Plot",
                icon="show_chart",
                value=True,
            ).classes("w-full"):
                await MigrationPlot().render()
            with ui.expansion(
                "Depth Sections",
                icon="vertical_align_center",
                value=True,
            ).classes("w-full"):
                await DepthSection(direction="north-south").render()
                await DepthSection(direction="east-west").render()

        if has_magnitude:
            ui.label("Magnitudes").classes("text-h2")
            with ui.row().classes("items-center gap-4 w-full"):
                await MagnitudeFrequency().render()
                await MagnitudeSemblance().render()
                await MagnitudeRate().render()
