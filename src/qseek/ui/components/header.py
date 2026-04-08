from __future__ import annotations

from pathlib import Path

from nicegui import ui

from qseek.ui.components.run_dialog import run_selection_dialog
from qseek.ui.components.search import EventSearch
from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state

_LOGO_SVG = (Path(__file__).parent.parent / "static" / "logo_light.svg").read_text()

_NAV_ITEMS = [
    ("Overview", "/"),
]


class Header:
    async def render(self, manager: SourceManager) -> None:
        with (
            ui.header()
            .classes("justify-center items-center px-4 gap-1")
            .style("background-color: #1a2338")
            .props("elevated"),
            ui.row().classes("w-full").style("max-width: 1290px"),
        ):
            with (
                ui.element("a")
                .props(
                    "href='https://pyrocko.github.io/qseek/' target='_blank' rel='noopener'"
                )
                .classes()
            ):
                ui.html(_LOGO_SVG, sanitize=False).classes("w-24")
            ui.button(
                "Overview",
                on_click=lambda _: ui.navigate.to("/"),
            ).props("flat color=white")
            catalog = await get_tab_state().run.get_catalog()
            if catalog.has_magnitudes:
                ui.button(
                    "Magnitudes",
                    on_click=lambda _: ui.navigate.to("/magnitudes"),
                ).props("flat color=white")
            else:
                ui.button("No magnitudes").props("flat color=grey-3 disable").classes(
                    "text-sm font-semibold px-3 mt1.0"
                )
            ui.space()

            await EventSearch().render()

            active_run = get_tab_state().run
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label(active_run.name).classes(
                    "text-white text-sm font-mono opacity-70 max-w-48 ellipsis"
                ).tooltip(active_run.name)
                ui.button(
                    icon="folder_open",
                    on_click=lambda: run_selection_dialog(manager),
                ).props("flat color=white round dense")
