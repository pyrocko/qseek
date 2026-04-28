from nicegui import ui

from qseek.ui.components.network import (
    StationCoverage,
    StationMap,
    StationsPickPerformance,
    StationTable,
    StationTraveltimeResiduals,
)


async def network_page() -> None:
    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            station_map = StationMap()
            station_map.header()
            await station_map.view()
            station_table = StationTable()
            await station_table.view()

        with ui.card().classes("w-full"):
            station_coverage = StationCoverage()
            station_coverage.header()
            await station_coverage.view()

        with ui.card().classes("w-full"):
            station_picks = StationsPickPerformance()
            station_picks.header()
            await station_picks.view()

        with ui.card().classes("w-full"):
            residuals = StationTraveltimeResiduals()
            residuals.header()
            await residuals.view()
