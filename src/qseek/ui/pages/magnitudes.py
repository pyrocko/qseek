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

        with ui.row().classes("items-center gap-4 w-full"):
            await magnitude_components.MagnitudeRate().render()

        with ui.row().classes("items-start gap-4 w-full"):
            with ui.column().classes("w-[calc(50%-0.5rem)]"):
                await magnitude_components.MagnitudeFrequency().render()
            with ui.column().classes("w-[calc(50%-0.5rem)]"):
                await magnitude_components.MagnitudeSemblance().render()
        with ui.row().classes("items-start gap-4 w-full"):
            await magnitude_components.StationMagnitudesOverStation().render()
            await magnitude_components.StationMagnitudeOverDistance().render()
