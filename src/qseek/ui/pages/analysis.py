from nicegui import ui

from qseek.ui.components.statistics import NPicksDistribution, WadatiDiagram


async def analysis_page() -> None:
    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12 col-md h-128"):
            wadati_plot = WadatiDiagram()
            wadati_plot.header()
            await wadati_plot.view()

        with ui.card().classes("col-12 col-md h-128"):
            npicks_dist = NPicksDistribution()
            npicks_dist.header()
            await npicks_dist.view()
