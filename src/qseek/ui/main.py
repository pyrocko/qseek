from __future__ import annotations

from pathlib import Path

from nicegui import ui

from qseek.search import Search
from qseek.ui.base import FrontendPage, Site
from qseek.ui.components.catalog import CatalogView  # noqa
from qseek.ui.components.event import EventDetails  # noqa


class Home(FrontendPage):
    path = "/"
    name = "Home"
    icon = "home"

    def ui(self) -> None:
        ui.label("Hello Home")


def start_ui(rundir: Path):
    search = Search.load_rundir(rundir)

    @ui.page("/")
    @ui.page("/{_:path}")
    def main() -> None:
        # ui.dark_mode(True)
        Site(search)

    ui.page_title("QSeek")
    ui.run(uvicorn_logging_level="info")


if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=Path)
    args = parser.parse_args()
    start_ui(args.rundir)
