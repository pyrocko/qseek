import numpy as np
from nicegui import ui

from qseek.ui.components.map import OverviewMap
from qseek.ui.components.network import (
    StationCoverage,
    StationsPickPerformance,
    StationTable,
    StationTraveltimeResiduals,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import card_header


async def network_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()
    search = await state.run.get_search()
    stations = search.stations
    events = catalog.full_catalog.events

    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            ui.label("Network Stations").classes("text-lg font-bold")
            ui.label(f"{stations.n_stations} stations available for search")
            station_map = OverviewMap(
                center_lat=np.mean([sta.lat for sta in stations]),
                center_lon=np.mean([sta.lon for sta in stations]),
            )
            await station_map.initialized()
            await station_map.add_stations(stations)

            StationTable(stations)

        with ui.card().classes("w-full"):
            card_header(StationCoverage.title, StationCoverage.description)
            station_coverage = StationCoverage()
            await station_coverage.plot_coverage(events)

        with ui.card().classes("w-full"):
            card_header(
                StationsPickPerformance.title, StationsPickPerformance.description
            )
            station_picks = StationsPickPerformance()
            await station_picks.plot_stations(events)

        with ui.card().classes("w-full"):
            card_header(
                StationTraveltimeResiduals.title, StationTraveltimeResiduals.description
            )
            residuals = StationTraveltimeResiduals()
            await residuals.plot_residuals(events)
