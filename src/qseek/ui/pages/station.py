from nicegui import ui

from qseek.ui.components.station import (
    StationDetails,
    StationMap,
    StationPickPerformance,
    StationTraveltimeResidual,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import StatCard, card_header


async def station_page(station_nsl: str) -> None:
    state = get_tab_state()
    search = await state.run.get_search()
    station = next(
        (
            sta
            for sta in search.stations
            if sta.nsl.pretty_str(strip=True) == station_nsl
        ),
        None,
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
        ui.label("Station Details").classes("text-h5")

    ui.separator().classes("mb-4")

    # Stat cards
    with ui.row().classes("w-full items-stretch"):
        StatCard(
            "Code",
            f"{station.nsl.pretty_str(strip=True)}",
            "sensors",
            tooltip="Network-Station-Location code (NSL)",
        )
        StatCard(
            "Latitude",
            f"{station.effective_lat:.5f}°",
            "explore",
        )
        StatCard(
            "Longitude",
            f"{station.effective_lon:.5f}°",
            "explore",
        )
        StatCard(
            "Elevation",
            f"{station.elevation:,.0f} m",
            "terrain",
            subtitle=f"Effective: {station.effective_elevation:,.0f} m",
        )
        if station.depth > 0:
            StatCard(
                "Depth",
                f"{station.depth:,.0f} m",
                "vertical_align_bottom",
            )

    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            card_header(
                StationMap.title,
                StationMap.description,
            )
            station_map = StationMap(station)
            await station_map.plot()

        with ui.card().classes("w-full"):
            card_header(
                StationDetails.title,
                StationDetails.description,
            )
            details = StationDetails(station)
            await details.plot()

        with ui.card().classes("w-full"):
            card_header(
                StationTraveltimeResidual.title,
                StationTraveltimeResidual.description,
            )
            station_residual = StationTraveltimeResidual(station)
            await station_residual.plot()

        with ui.card().classes("w-full"):
            card_header(
                StationPickPerformance.title,
                StationPickPerformance.description,
            )
            pick_perf = StationPickPerformance(station)
            await pick_perf.plot()
