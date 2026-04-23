from nicegui import ui

from qseek.ui.components.network import StationMap, StationTable


async def network_page() -> None:
    with ui.column().classes("w-full gap-4"):
        with ui.card().classes("w-full"):
            station_map = StationMap()
            station_map.header()
            await station_map.view()
        with ui.card().classes("w-full"):
            station_table = StationTable()
            station_table.header()
            await station_table.view()
