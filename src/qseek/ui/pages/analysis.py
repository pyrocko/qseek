from nicegui import background_tasks, ui

from qseek.ui.components.statistics import (
    NPicksDistribution,
    SemblanceDistribution,
    WadatiDiagram,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import card_header


async def analysis_page() -> None:
    catalog = await get_tab_state().get_catalog()
    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12 col-md"):
            card_header(WadatiDiagram.title, WadatiDiagram.description)
            wadati_plot = WadatiDiagram()
            background_tasks.create(wadati_plot.plot_events(catalog.events))

        with ui.card().classes("col-12 col-md"):
            card_header(NPicksDistribution.title, NPicksDistribution.description)
            npicks_dist = NPicksDistribution()
            background_tasks.create(npicks_dist.plot_picks(catalog.events))

    with (
        ui.row().classes("w-full flex-1 items-stretch"),
        ui.card().classes("col-12 col-md h-128"),
    ):
        card_header(SemblanceDistribution.title, SemblanceDistribution.description)
        semblance_dist = SemblanceDistribution()
        background_tasks.create(semblance_dist.plot_distribution(catalog.events))
