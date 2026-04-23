from __future__ import annotations

from nicegui import ui

from qseek.ui.components import magnitudes as magnitude_components
from qseek.ui.state import get_tab_state


async def magnitudes_page() -> None:
    state = get_tab_state()
    catalog = await state.get_filtered_catalog()

    if not catalog.has_magnitudes():
        with ui.column().classes("w-full items-center justify-center gap-3 mt-16"):
            ui.icon("bar_chart").classes("text-grey-5").style("font-size: 4rem")
            ui.label("No magnitudes available").classes("text-h6 text-grey-6")
            ui.label(
                "No event magnitudes are present in this catalog yet. "
                "Run magnitude estimation to enable these plots."
            ).classes("text-grey-5 text-body2 text-center").style("max-width: 380px")
        return

    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12"):
            rate = magnitude_components.MagnitudeRate()
            rate.header()
            await rate.view()

        with ui.card().classes("col-12 col-md"):
            freq = magnitude_components.MagnitudeFrequency()
            freq.header()
            await freq.view()

        with ui.card().classes("col-12 col-md"):
            semblance = magnitude_components.MagnitudeSemblance()
            semblance.header()
            await semblance.view()

        with ui.card().classes("col-12 col-md"):
            over_station = magnitude_components.StationMagnitudesOverStation()
            over_station.header()
            await over_station.view()

        with ui.card().classes("col-12 col-md"):
            over_distance = magnitude_components.StationMagnitudeOverDistance()
            over_distance.header()
            await over_distance.view()
