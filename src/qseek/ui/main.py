import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from fastapi import Response
from nicegui import app, core, ui

from qseek.utils import load_insights, setup_rich_logging

load_insights()

_LOGO_SVG = (Path(__file__).parent / "static" / "logo_light.svg").read_text()


def start_ui(uris: list[str], reload: bool = True, port: int = 2251) -> None:
    from qseek.ui.layout import drawer, header
    from qseek.ui.manager import SourceManager
    from qseek.ui.pages.analysis import analysis_page
    from qseek.ui.pages.clusters import clusters_page
    from qseek.ui.pages.config import config_page
    from qseek.ui.pages.event import event_page
    from qseek.ui.pages.magnitudes import magnitudes_page
    from qseek.ui.pages.network import network_page
    from qseek.ui.pages.overview import overview_page
    from qseek.ui.pages.station import station_page
    from qseek.ui.state import create_tab_state

    manager = SourceManager()
    ready = asyncio.Event()

    async def load_runs():
        core.sio.eio.ping_interval = 30.0
        core.sio.eio.ping_timeout = 30.0
        await manager.add_uris(uris)

        if manager.n_runs == 0:
            app.shutdown()
            raise EnvironmentError("No runs found at the specified URIs")
        ready.set()

    app.add_static_files("/static", Path(__file__).parent / "static")
    app.on_startup(load_runs)

    @ui.page("/run/{run_id}/csv")
    async def csv_page(run_id: str) -> None | Response:
        try:
            run = manager.get_run(run_id)
        except KeyError:
            ui.status_code(404)
            ui.label("Run not found").classes("text-red-500")
            return None

        catalog = await run.get_catalog()
        with NamedTemporaryFile() as tmp:
            await catalog.export_csv(tmp.name)
            tmp.seek(0)
            return Response(
                content=tmp.read(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={run.name}.csv"},
            )

    @ui.page("/run/{run_id}/gpkg")
    async def gpkg_page(run_id: str) -> None | Response:
        try:
            run = manager.get_run(run_id)
        except KeyError:
            ui.status_code(404)
            ui.label("Run not found").classes("text-red-500")
            return None

        catalog = await run.get_catalog()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "catalog.gpkg"
            await catalog.export_geopackage(tmp_path)
            return Response(
                content=tmp_path.read_bytes(),
                media_type="application/geopackage+sqlite3",
                headers={
                    "Content-Disposition": f"attachment; filename={run.name}.gpkg"
                },
            )

    @ui.page("/")
    @ui.page("/{_:path}")
    async def main_page() -> None:
        await ready.wait()
        await ui.context.client.connected()

        state = create_tab_state(run=manager.get_default_run())
        drawer(manager)
        await header()

        with (
            ui.row()
            .classes("w-full items-stretch mx-auto")
            .style("max-width: 1290px; min-height: 85vh")
        ):
            with (
                ui.column().classes(
                    "flex-1 gap-3 items-center justify-center text-center text-lg"
                ) as column,
                ui.card()
                .classes("rounded-lg shadow-2 items-center justify-center p-8")
                .style("background-color: #1a2a3f"),
            ):
                ui.html(_LOGO_SVG, sanitize=False).classes("w-32")
                ui.label().bind_text_from(state, "loading").classes(
                    "text-lg font-medium mt-2 text-white"
                )
                column.bind_visibility_from(state, "loading")

            sub_pages = (
                ui.sub_pages(
                    {
                        "/": overview_page,
                        "/magnitudes": magnitudes_page,
                        "/analysis": analysis_page,
                        "/network": network_page,
                        "/config": config_page,
                        "/event/{event_id}": event_page,
                        "/station/{station_nsl}": station_page,
                        "/clusters": clusters_page,
                    },
                    show_404=False,
                )
                .classes("flex-grow")
                .bind_visibility_from(state, "loading", backward=lambda v: not v)
            )

        async def on_run_changed():
            ui.navigate.to("/")
            sub_pages.refresh()

        state.run_changed.subscribe(on_run_changed)
        await ui.context.client.disconnected()
        await state.clear()

    ui.run(
        title="Qseek Explorer",
        favicon="🚀",
        port=port,
        reload=reload,
        uvicorn_logging_level="debug",
    )


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
    start_ui(args.uris, reload=True)
