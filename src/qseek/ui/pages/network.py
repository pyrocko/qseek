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


async def network_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()
    search = await state.run.get_search()
    stations = search.stations
    events = catalog.full_catalog.events

    with ui.column().classes("w-full gap-4"):
        map_ = OverviewMap(
            center_lat=np.mean([sta.lat for sta in stations]),
            center_lon=np.mean([sta.lon for sta in stations]),
        )
        map_.set_title("Network Stations")
        map_.set_description(f"{stations.n_stations} stations available for search")
        await map_.initialize()
        await map_.add_stations(stations)
        await map_.update_extent()

        StationTable(stations)

        station_coverage = StationCoverage()
        await station_coverage.plot_coverage(events)
        station_coverage.attach_catalog(catalog)

        station_picks = StationsPickPerformance()
        await station_picks.plot_stations(events)

        residuals = StationTraveltimeResiduals()
        await residuals.plot_residuals(events)
