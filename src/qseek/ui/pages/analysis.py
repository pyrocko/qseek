from nicegui import background_tasks, ui

from qseek.ui.components.statistics import (
    NPicksDistribution,
    SemblanceDistribution,
    WadatiDiagram,
)
from qseek.ui.state import get_tab_state


async def analysis_page() -> None:
    catalog = await get_tab_state().get_catalog()
    with ui.row().classes("w-full flex-1 items-stretch"):
        wadati_plot = WadatiDiagram().classes("col-12 col-md")
        background_tasks.create(wadati_plot.plot_events(catalog.events))

        npicks_dist = NPicksDistribution().classes("col-12 col-md")
        background_tasks.create(npicks_dist.plot_picks(catalog.events))

    with (
        ui.row().classes("w-full flex-1 items-stretch"),
    ):
        semblance_dist = SemblanceDistribution()
        background_tasks.create(semblance_dist.plot_distribution(catalog.events))
