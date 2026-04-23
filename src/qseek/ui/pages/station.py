from nicegui import ui

from qseek.ui.components.station import StationDetails, StationMap
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


async def station_page(station_nsl: str) -> None:
    state = get_tab_state()
    search = await state.run.get_search()
    station = next(
        (sta for sta in search.stations if sta.nsl.pretty == station_nsl), None
    )

    if station is None:
        with ui.column().classes("w-full items-center gap-4 mt-16"):
            ui.icon("sensors_off").classes("text-6xl text-grey-4")
            ui.label(f"Station '{station_nsl}' not found in inventory.").classes(
                "text-grey-6 text-lg"
            )
            ui.button("Go back", icon="arrow_back", on_click=ui.navigate.back).props(
                "flat"
            )
        return

    # Header
    with ui.row().classes("w-full items-center gap-2 mb-1"):
        ui.button(icon="arrow_back", on_click=ui.navigate.back).props("flat round")
        ui.icon("sensors").classes("text-grey-7 text-2xl")
        ui.label(station.nsl.pretty).classes("text-h5 font-mono")

    ui.separator().classes("mb-4")

    # Stat cards
    with ui.row().classes("w-full items-stretch"):
        stat_card(
            "Network",
            station.network,
            "hub",
        )
        stat_card(
            "Station",
            station.station,
            "sensors",
        )
        stat_card(
            "Location",
            station.location or "—",
            "label",
        )
        stat_card(
            "Latitude",
            f"{station.effective_lat:.5f}°",
            "explore",
        )
        stat_card(
            "Longitude",
            f"{station.effective_lon:.5f}°",
            "explore",
        )
        stat_card(
            "Elevation",
            f"{station.elevation:,.0f} m",
            "terrain",
            subtitle=f"Effective: {station.effective_elevation:,.0f} m",
        )
        if station.depth > 0:
            stat_card(
                "Depth",
                f"{station.depth:,.0f} m",
                "vertical_align_bottom",
            )

    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            station_map = StationMap(station)
            station_map.header()
            await station_map.view()

        with ui.card().classes("w-full"):
            details = StationDetails(station)
            details.header()
            await details.view()
