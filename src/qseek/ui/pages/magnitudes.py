from __future__ import annotations

from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components import magnitudes as magnitude_components
from qseek.ui.state import get_tab_state


class MagnitudesPage(Page):
    async def render(self) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()

        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label("Magnitudes").classes("text-h1")

        if not catalog.has_magnitudes():
            with ui.row().classes("items-center gap-2 text-grey-6 mt-2"):
                ui.icon("info").classes("text-grey-6")
                ui.label("No magnitudes available").classes("text-body1 font-medium")
            ui.label(
                "No event magnitudes are present in this catalog yet. "
                "Run magnitude estimation to enable these plots."
            ).classes("text-grey-6 text-body2")
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
