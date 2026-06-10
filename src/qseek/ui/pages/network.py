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
    catalog = await get_tab_state().get_catalog()

    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            station_map = OverviewMap(catalog)
            station_map.header(
                title="Network Overview",
                description="Map of stations. Click on a station to view its details.",
            )
            await station_map.view(show_events=False)
            station_table = StationTable(catalog)
            await station_table.view()

        with ui.card().classes("w-full"):
            station_coverage = StationCoverage(catalog)
            station_coverage.header()
            await station_coverage.view()

        with ui.card().classes("w-full"):
            station_picks = StationsPickPerformance(catalog)
            station_picks.header()
            await station_picks.view()

        with ui.card().classes("w-full"):
            residuals = StationTraveltimeResiduals(catalog)
            residuals.header()
            await residuals.view()
