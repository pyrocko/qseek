from __future__ import annotations

from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components import magnitudes as magnitude_components
from qseek.ui.state import get_tab_state


class MagnitudesPage(Page):
    async def render(self) -> None:
        run = get_tab_state().run
        catalog = await run.get_catalog()

        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label("Magnitudes").classes("text-h1")
            ui.space()
            ui.chip(
                str(
                    f"Time range: {catalog.times[0].date()} - {catalog.times[-1].date()}"
                ),
                icon="schedule",
            ).props("outline").classes("text-xs font-mono text-grey-9")

        if not catalog.has_magnitudes:
            ui.label("No magnitudes").classes("text-grey-6 text-body1")
            return

        with ui.row().classes("items-center gap-4 w-full"):
            await magnitude_components.MagnitudeFrequency().render()
            await magnitude_components.MagnitudeSemblance().render()
            await magnitude_components.MagnitudeRate().render()
