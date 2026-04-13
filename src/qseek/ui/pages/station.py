from __future__ import annotations

from urllib.parse import unquote

from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.station import (
    StationActiveTimeRangesPlot,
    StationMagnitudeResidualHistogramPlot,
    StationMagnitudeResidualTimePlot,
    StationPickResidualHistogramPlot,
    StationSNRTimePlot,
    collect_station_page_data,
)
from qseek.ui.state import get_tab_state
from qseek.utils import _NSL


class StationPage(Page):
    async def render(self, station_id: str) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()

        station_nsl = _NSL.parse(unquote(station_id))
        data = collect_station_page_data(catalog, station_nsl)
        station_label = data.station_label

        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.button(icon="arrow_back", on_click=ui.navigate.back).props("flat round")
            ui.label(f"Station {station_label}").classes("text-h5 font-mono")
            ui.space()
            ui.chip(station_label, icon="sensors").props("outline").classes(
                "text-xs font-mono text-grey-6"
            )

        ui.separator().classes("mb-4")

        if not data.event_rows:
            with ui.row().classes("items-center gap-2 text-grey-6 mt-2"):
                ui.icon("info").classes("text-grey-6")
                ui.label("Station not present in filtered catalog").classes(
                    "text-body1 font-medium"
                )
            ui.label(
                "Adjust filters or select another station to see station diagnostics."
            ).classes("text-grey-6 text-body2")
            return

        await StationActiveTimeRangesPlot(data).render()

        if not data.station_mag_rows:
            with ui.row().classes("w-full items-start gap-4"):
                await StationPickResidualHistogramPlot(data).render()
            with ui.row().classes("items-center gap-2 text-grey-6 mt-2"):
                ui.icon("info").classes("text-grey-6")
                ui.label("No station magnitudes for this station").classes(
                    "text-body1 font-medium"
                )
            return

        with ui.row().classes("w-full items-start gap-4"):
            await StationPickResidualHistogramPlot(data).render()
            await StationSNRTimePlot(data).render()

        with ui.row().classes("w-full items-start gap-4"):
            await StationMagnitudeResidualHistogramPlot(data).render()
            await StationMagnitudeResidualTimePlot(data).render()
