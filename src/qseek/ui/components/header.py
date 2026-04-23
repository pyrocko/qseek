from __future__ import annotations

from nicegui import ui

from qseek.ui.components.search import event_search


async def header() -> None:
    with (
        ui.header()
        .classes("justify-center items-center px-4 gap-1")
        .style("background-color: white"),
        ui.row().classes("w-full").style("max-width: 1290px"),
    ):
        ui.space()
        await event_search()
        ui.space()
