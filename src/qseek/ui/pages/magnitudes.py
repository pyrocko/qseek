from __future__ import annotations

from nicegui import ui

from qseek.ui.components.magnitudes import (
    MagnitudeFrequency,
    MagnitudeFrequencyBPositive,
    MagnitudeRate,
    MagnitudeStatisticsOverTime,
    StationsMagnitudesResiduals,
)
from qseek.ui.state import get_tab_state


async def magnitudes_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()
    events = catalog.events

    if not catalog.has_magnitudes():
        with ui.column().classes("w-full items-center justify-center gap-3 mt-16"):
            ui.icon("bar_chart").classes("text-grey-5").style("font-size: 4rem")
            ui.label("No magnitudes available").classes("text-h6 text-grey-6")
            ui.label(
                "No event magnitudes are present in this catalog yet. "
                "Run magnitude estimation to enable these plots."
            ).classes("text-grey-5 text-body2 text-center").style("max-width: 380px")
        return

    mag_rate = MagnitudeRate(
        show_semblance=not catalog.has_magnitudes(),
        show_density=True,
    )
    await mag_rate.plot_events(events)

    stats_over_time = MagnitudeStatisticsOverTime()
    await stats_over_time.plot_events(events)

    with ui.row().classes("w-full flex-1 items-stretch"):
        freq = MagnitudeFrequency()
        await freq.plot_events(events)

        with ui.card().classes("col-12 col-md"):
            b_positive = MagnitudeFrequencyBPositive()
            await b_positive.plot_events(events)

    with ui.row().classes("w-full flex-1 items-stretch"):
        over_station = StationsMagnitudesResiduals()
        await over_station.plot_residuals(events)
