from __future__ import annotations

from pathlib import Path

from nicegui import ui

from qseek.ui.components.event_search import EventSearch
from qseek.ui.components.run_manager import run_selection_dialog
from qseek.ui.models import RunManager

_LOGO_SVG = (Path(__file__).parent.parent / "static" / "logo_light.svg").read_text()

_NAV_ITEMS = [
    ("Overview", "/"),
]


class Header:
    def __init__(self, manager: RunManager) -> None:
        self.manager = manager

    def render(self) -> None:
        with (
            ui.header()
            .classes("justify-center items-center px-4 gap-1")
            .style("background-color: #1a2338")
            .props("elevated"),
            ui.row().classes("w-full").style("max-width: 1290px"),
        ):
            ui.html(_LOGO_SVG, sanitize=False).classes("w-24")
            ui.button(
                "Overview",
                on_click=lambda: ui.navigate("/"),
            ).props("flat color=white")
            ui.space()

            EventSearch().render()

            active_run = self.manager.get_active_run()
            ui.button(
                active_run.path.name,
                icon="folder_open",
                on_click=lambda: run_selection_dialog(self.manager),
            ).props("flat color=white")
