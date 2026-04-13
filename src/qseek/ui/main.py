import asyncio
import logging
from pathlib import Path

from nicegui import app, core, ui

from qseek.ui.components.header import Header
from qseek.ui.pages.event import EventPage
from qseek.ui.pages.magnitudes import MagnitudesPage
from qseek.ui.pages.network import NetworkPage as NetworkPage
from qseek.ui.pages.overview import OverviewPage
from qseek.ui.pages.station import StationPage
from qseek.ui.state import TabState, get_tab_state
from qseek.utils import load_insights, setup_rich_logging

load_insights()

from qseek.ui.manager import SourceManager  # noqa: E402

ui.card.default_classes = "flex-1 min-w-md col-6 shadow-2"


def start_ui(uris: list[str], reload: bool = True) -> None:
    app.add_static_files("/static", Path(__file__).parent / "static")

    manager = SourceManager()
    ready = asyncio.Event()

    async def load_runs():
        core.sio.eio.ping_interval = 30.0
        core.sio.eio.ping_timeout = 30.0
        await manager.add_uris(uris)

        if manager.n_runs == 0:
            app.shutdown()
            raise EnvironmentError("No runs found at the specified URIs")
        default_run = list(manager.runs.values())[1]
        TabState.set_default_run(default_run)
        ready.set()

    app.on_startup(load_runs)

    @ui.page("/")
    @ui.page("/{_:path}")
    async def main_page() -> None:
        await ready.wait()
        await ui.context.client.connected()

        state = get_tab_state()

        await Header().render(manager)

        with ui.row().classes("w-full").style("max-width: 1290px").classes("mx-auto"):
            with ui.column().classes(
                "gap-3 w-full items-center justify-center text-center"
                " text-lg pt-40 pb-40 opacity-60"
            ) as column:
                ui.spinner("audio", size="lg", color="gray")
                ui.label().bind_text_from(
                    state, "run_name", backward=lambda n: f"Loading run {n}..."
                )
                column.bind_visibility_from(state, "loading")

            sub_pages = (
                ui.sub_pages(
                    {
                        "/": OverviewPage().render,
                        "/magnitudes": MagnitudesPage().render,
                        "/event/{event_id}": EventPage().render,
                        "/station/{station_id}": StationPage().render,
                        "/network": NetworkPage().render,
                    },
                    show_404=False,
                )
                .classes("flex-grow")
                .bind_visibility_from(state, "loading", backward=lambda v: not v)
            )

        with (
            ui.row().classes("items-center opacity-60 px-4 py-3 w-full justify-center"),
            ui.column().classes("items-center"),
        ):
            ui.label(
                "Qseek - The Earthquake Detection and Localization Framework 🔥"
            ).classes("text-darkgray text-s")
            ui.html(
                '<a href="https://github.com/pyrocko/qseek" target="_blank" rel="noopener" style="display:flex;align-items:center"><svg height="30" viewBox="0 0 16 16" width="30" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" fill="black"/></svg></a>',
                sanitize=False,
            )

        async def on_run_changed():
            ui.navigate.to("/")
            sub_pages.refresh()

        state.filtered_catalog.updated.subscribe(on_run_changed)

    ui.run(title="Qseek Explorer", favicon="🚀", reload=reload)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uris",
        type=str,
        nargs="+",
        help="Path or URI to Qseek runs"
        " (e.g. /path/to/runs or ssh://user@host:port/path/to/runs)",
    )
    args = parser.parse_args()
    setup_rich_logging(level=logging.INFO)
    start_ui(args.uris)
