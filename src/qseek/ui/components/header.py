from __future__ import annotations

from nicegui import ui

from qseek.ui.components.run_manager import run_selection_dialog
from qseek.ui.models import RunManager

_NAV_ITEMS = [
    ("Overview", "/"),
    ("Statistics", "/statistics"),
    ("Network", "/network"),
    ("Event Details", "/event-details"),
]


class Header:
    def __init__(self, manager: RunManager) -> None:
        self.manager = manager

    def render(self) -> None:
        dark = ui.dark_mode()

        with (
            ui.header()
            .classes("justify-center items-center px-4 gap-1 bg-grey-10")
            .props("elevated"),
            ui.row().classes("w-full").style("max-width: 1290px"),
        ):
            ui.image("/static/logo_light.svg").classes("w-24")
            ui.separator().props("vertical").classes("opacity-20 my-2")
            for label, path in _NAV_ITEMS:
                ui.button(
                    label,
                    on_click=lambda _, p=path: ui.navigate.to(p),
                ).props("flat color=white")
            ui.space()
            ui.separator().props("vertical").classes("opacity-20 my-2")

            active_run = self.manager.get_active_run()
            ui.button(
                active_run.path.name,
                icon="folder_open",
                on_click=lambda: run_selection_dialog(self.manager),
            ).props("flat color=white")

            ui.button(
                icon="brightness_6",
                on_click=dark.toggle,
            ).props("flat round color=white")
