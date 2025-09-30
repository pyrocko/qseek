import logging
from pathlib import Path

from nicegui import app, ui

from qseek.ui.header import Header
from qseek.ui.overview import OverviewPage
from qseek.utils import load_insights, setup_rich_logging

load_insights()

from qseek.ui.models import RunManager  # noqa: E402


def start_ui(basepath: Path, reload: bool = True) -> None:
    app.add_static_files("/static", Path(__file__).parent / "static")

    manager = RunManager()
    manager.add_dir(basepath)

    if not manager.n_runs:
        raise RuntimeError(f"No runs found in {basepath}")

    @ui.page("/")
    @ui.page("/{_:path}")
    def main_page():
        header = Header(manager)
        header.render()

        overview_page = OverviewPage(manager)

        with ui.row().classes("w-full").style("max-width: 2100px").classes("mx-auto"):
            pages = ui.sub_pages(
                {
                    "/": overview_page.render,
                    # "/statistics/{run_hash}": statistics,
                    # "/network/{run_hash}": network,
                    # "/event-details/{run_hash}/{event_id}": event_details,
                }
            ).classes("flex-grow p-4")
        manager.on_active_run_change(pages.refresh)

    ui.run(title="Qseek Explorer", reload=reload)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("basepath", type=Path)
    args = parser.parse_args()
    setup_rich_logging(level=logging.INFO)
    start_ui(args.basepath)
