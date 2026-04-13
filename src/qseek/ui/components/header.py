from __future__ import annotations

from pathlib import Path

from nicegui import ui

from qseek.ui.components.catalog_filter import catalog_filter_dialog
from qseek.ui.components.run_dialog import run_selection_dialog
from qseek.ui.components.search import EventSearch
from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state

_LOGO_SVG = (Path(__file__).parent.parent / "static" / "logo_light.svg").read_text()


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
            ui.button(
                "Magnitudes",
                on_click=lambda _: ui.navigate.to("/magnitudes"),
            ).props("flat color=white")
            ui.button(
                "Network",
                on_click=lambda _: ui.navigate.to("/network"),
            ).props("flat color=white")
            ui.space()

            await EventSearch().render()

            tab_state = get_tab_state()
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label().classes(
                    "text-white text-sm font-mono opacity-70 max-w-48 ellipsis"
                ).bind_text_from(tab_state, "run_name").tooltip(tab_state.run_name)
                ui.button(
                    icon="folder_open",
                    on_click=lambda: run_selection_dialog(manager),
                ).props("flat color=white round dense")

                ui.button(
                    icon="filter_alt",
                    on_click=catalog_filter_dialog,
                ).props("flat color=white round dense")
