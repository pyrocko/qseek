from nicegui import ui

from qseek.ui.components.statistics import (
    NPicksDistribution,
    SemblanceDistribution,
    WadatiDiagram,
)
from qseek.ui.state import get_tab_state


async def analysis_page() -> None:
    catalog = await get_tab_state().get_catalog()
    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.row():
            with ui.card().classes("col-12 col-md h-128"):
                wadati_plot = WadatiDiagram(catalog)
                wadati_plot.header()
                await wadati_plot.view()

            with ui.card().classes("col-12 col-md h-128"):
                npicks_dist = NPicksDistribution(catalog)
                npicks_dist.header()
                await npicks_dist.view()

        with ui.card().classes("col-12 col-md h-128"):
            semblance_dist = SemblanceDistribution(catalog)
            semblance_dist.header()
            await semblance_dist.view()
