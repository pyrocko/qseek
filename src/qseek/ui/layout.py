from pathlib import Path

from nicegui import ui

from qseek.ui.components.catalog_filter import catalog_filter_dialog
from qseek.ui.components.run_dialog import run_selection_dialog
from qseek.ui.components.search import event_search
from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state

_LOGO_SVG = (Path(__file__).parent / "static" / "logo_light.svg").read_text()


async def header() -> None:
    with (
        ui.header()
        .classes("justify-center items-center px-4 gap-1 bg-white/30 backdrop-blur-sm")
        .props("bordered"),
        ui.row().classes("w-full").style("max-width: 1290px"),
    ):
        ui.space()
        await event_search()
        ui.space()


def _drawer_button(name: str, icon: str, link: str) -> None:
    with ui.item().on_click(lambda: ui.navigate.to(link)).props("ripple"):
        with ui.item_section().props("avatar"):
            ui.icon(icon).classes("text-gray-200")
        with ui.item_section():
            ui.item_label(name).props("strong").classes("text-white text-lg")


def drawer(manager: SourceManager) -> None:
    tab_state = get_tab_state()

    with (
        ui.left_drawer(top_corner=True, bottom_corner=True)
        .classes("w-40")
        .style("background-color: #1a2a3f")
        .props("elevated"),
    ):
        with (
            ui.element("a")
            .props(
                "href='https://pyrocko.github.io/qseek/' target='_blank' rel='noopener'"
            )
            .classes("p-2 pb-2")
        ):
            ui.html(_LOGO_SVG, sanitize=False).classes("w-32")

        with ui.list().classes("w-full"):
            _drawer_button("Overview", "dashboard", "/")
            _drawer_button("Network", "hub", "/network")
            _drawer_button("Analysis", "analytics", "/analysis")
            _drawer_button("Events", "crisis_alert", "/events")
            _drawer_button("Magnitudes", "equalizer", "/magnitudes")

        ui.space()

        with ui.button_group().classes("w-full"):
            ui.button(
                icon="folder_open",
                on_click=lambda: run_selection_dialog(manager),
            ).bind_text_from(tab_state, "run_name").classes("w-full")
            ui.button(
                icon="filter_alt",
                on_click=catalog_filter_dialog,
            ).props("outline")
        ui.separator()
        with (
            ui.row().classes("items-center opacity-60 p-2 w-full justify-center"),
            ui.column().classes("items-center"),
        ):
            ui.label("Qseek - Earthquake Detection and Localization").classes(
                "text-white text-s text-center"
            )
            ui.html(
                '<a href="https://github.com/pyrocko/qseek" target="_blank" rel="noopener" style="display:flex;align-items:center"><svg height="30" viewBox="0 0 16 16" width="30" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" fill="lightgray"/></svg></a>',
                sanitize=False,
            )
