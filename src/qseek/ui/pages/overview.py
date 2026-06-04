from __future__ import annotations

import numpy as np
from nicegui import ui

from qseek.ui.components.magnitudes import MagnitudeRate
from qseek.ui.components.map import OverviewMap
from qseek.ui.components.statistics import (
    EventRate,
    SemblanceRate,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


async def overview_page() -> None:
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

    rms_values = np.array(
        [ev.event.rms for ev in catalog.events if ev.event.rms is not None],
        dtype=float,
    )
    has_rms = rms_values.size > 0

    if catalog.has_magnitudes():
        magnitude_max = np.nanmax(catalog.magnitudes)
        magnitude_min = np.nanmin(catalog.magnitudes)
    else:
        magnitude_max = 0.0
        magnitude_min = 0.0

    event_rate = len(catalog.events) / (
        (catalog.times[-1] - catalog.times[0]).total_seconds() / (3600 * 24)
    )
    num_days = int((catalog.times[-1] - catalog.times[0]).total_seconds() / (3600 * 24))

    with ui.row().classes("w-full items-stretch"):
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
            subtitle=f"Min {n_picks_min} / Max {n_picks_max}",
            tooltip="Median number of picks per event.",
        )
        if has_rms:
            rms_median_ms = float(np.nanmedian(rms_values)) * 1e3
            rms_min_ms = float(np.nanmin(rms_values)) * 1e3
            rms_max_ms = float(np.nanmax(rms_values)) * 1e3
            stat_card(
                "Median RMS",
                f"{rms_median_ms:.1f} ms",
                icon="manage_history",
                subtitle=f"Min {rms_min_ms:.1f} / Max {rms_max_ms:.1f} ms",
                tooltip="Median RMS of traveltime delays across all detected events.",
            )
        if catalog.has_magnitudes():
            stat_card(
                "Max Magnitude",
                f"{magnitude_max:.2f}",
                icon="bar_chart",
                subtitle=f"Min Magnitude: {magnitude_min:.2f}",
                tooltip="Maximum magnitude among all detected events.",
            )

    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12"):
            overview = OverviewMap()
            overview.header()
            await overview.view()
        if catalog.has_magnitudes():
            with ui.card().classes("col-12"):
                rate = MagnitudeRate()
                rate.header()
                await rate.view()
        with ui.card().classes("col-12"):
            semblance_rate = SemblanceRate()
            semblance_rate.header()
            await semblance_rate.view()
        with ui.card().classes("col-12"):
            event_rate = EventRate()
            event_rate.header()
            await event_rate.view()
        # with ui.card().classes("col-12"):
        #     migration_plot = MigrationPlot()
        #     migration_plot.header()
        #     await migration_plot.view()
        # with ui.card().classes("col-12"):
        #     depth_section = DepthSection()
        #     depth_section.header()
        #     await depth_section.view(direction="north-south")
        #     await depth_section.view(direction="east-west")
