from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path

import nest_asyncio
from pkg_resources import get_distribution
from rich import box
from rich.prompt import IntPrompt
from rich.table import Table

from qseek.console import console
from qseek.utils import CACHE_DIR, import_insights, setup_rich_logging

nest_asyncio.apply()

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qseek",
        description="qseek - The wholesome earthquake detector 🚀",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="increase verbosity of the log messages, repeat to increase. "
        "Default level is INFO",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_distribution("qseek").version,
        help="show version and exit",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        required=True,
        dest="command",
        description="Available commands to run qseek. Get command help with "
        "`qseek <command> --help`.",
    )

    init_project = subparsers.add_parser(
        "init",
        help="initialize a new qseek project",
        description="initialze a new project with a default configuration file. ",
    )
    init_project.add_argument(
        "folder",
        type=Path,
        help="folder to initialize project in",
    )

    run = subparsers.add_parser(
        "search",
        help="start a search",
        description="detect, localize and characterize earthquakes in a dataset",
    )
    run.add_argument("config", type=Path, help="path to config file")
    run.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="backup old rundir and create a new",
    )

    continue_run = subparsers.add_parser(
        "continue",
        help="continue an aborted run",
        description="continue a run from an existing rundir",
    )
    continue_run.add_argument("rundir", type=Path, help="existing runding to continue")

    features = subparsers.add_parser(
        "feature-extraction",
        help="extract features from an existing run",
        description="modify the search.json for re-evaluation of the event's features",
    )
    features.add_argument("rundir", type=Path, help="path of existing run")

    station_corrections = subparsers.add_parser(
        "corrections",
        help="analyse station corrections from existing run",
        description="analyze and plot station corrections from a finished run",
    )
    station_corrections.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot station correction results and save to rundir",
    )
    station_corrections.add_argument("rundir", type=Path, help="path of existing run")

    subparsers.add_parser(
        "modules",
        help="list available modules",
        description="list all available modules",
    )

    serve = subparsers.add_parser(
        "serve",
        help="start webserver and serve results from an existing run",
        description="start a webserver and serve detections and results from a run",
    )
    serve.add_argument("rundir", type=Path, help="rundir to serve")

    subparsers.add_parser(
        "clear-cache",
        help="clear the cach directory",
        description="clear all data in the cache directory",
    )

    dump_schemas = subparsers.add_parser(
        "dump-schemas",
        help="dump data models to json-schema (development)",
        description="dump data models to json-schema, "
        "this is for development purposes only",
    )
    dump_schemas.add_argument("folder", type=Path, help="folder to dump schemas to")

    return parser


def main() -> None:
    import_insights()
    parser = get_parser()
    args = parser.parse_args()

    from qseek.models import Stations
    from qseek.search import Search
    from qseek.server import WebServer

    setup_rich_logging(level=logging.INFO - args.verbose * 10)

    match args.command:
        case "init":
            folder: Path = args.folder
            if folder.exists():
                raise FileExistsError(f"Folder {folder} already exists")
            folder.mkdir()

            pyrocko_stations = folder / "pyrocko-stations.yaml"
            pyrocko_stations.touch()

            config = Search(stations=Stations(pyrocko_station_yamls=[pyrocko_stations]))

            config_file = folder / f"{folder.name}.json"
            config_file.write_text(config.model_dump_json(by_alias=False, indent=2))

            logger.info("initialized new project in folder %s", folder)
            logger.info("start detection with: qseek run %s", config_file.name)

        case "search":
            search = Search.from_config(args.config)

            webserver = WebServer(search)

            async def run() -> None:
                http = asyncio.create_task(webserver.start())
                await search.start(force_rundir=args.force)
                await http

            asyncio.run(run())

        case "continue":
            search = Search.load_rundir(args.rundir)
            if search._progress.time_progress:
                console.rule(f"Continuing search from {search._progress.time_progress}")
            else:
                console.rule("Starting search from scratch")

            webserver = WebServer(search)

            async def run() -> None:
                http = asyncio.create_task(webserver.start())
                await search.start()
                await http

            asyncio.run(run())

        case "feature-extraction":
            search = Search.load_rundir(args.rundir)

            async def extract() -> None:
                for detection in search._detections.detections:
                    await search.add_features(detection)

            asyncio.run(extract())

        case "corrections":
            rundir = Path(args.rundir)
            from qseek.corrections.base import StationCorrections

            corrections_modules = StationCorrections.get_subclasses()

            console.print("[bold]Available travel time corrections modules")
            for imodule, module in enumerate(corrections_modules):
                console.print(f"{imodule}: {module.__name__}")

            module_choice = IntPrompt.ask(
                "Choose correction module",
                choices=[str(i) for i in range(len(corrections_modules))],
                default="0",
                console=console,
            )
            corrections = corrections_modules[int(module_choice)]
            asyncio.run(corrections.prepare(rundir, console))

        case "serve":
            search = Search.load_rundir(args.rundir)
            webserver = WebServer(search)

            loop = asyncio.get_event_loop()
            loop.create_task(webserver.start())
            loop.run_forever()

        case "clear-cache":
            logger.info("clearing cache directory %s", CACHE_DIR)
            shutil.rmtree(CACHE_DIR)

        case "modules":
            from qseek.corrections.base import StationCorrections
            from qseek.features.base import FeatureExtractor
            from qseek.tracers.base import RayTracer
            from qseek.waveforms.base import WaveformProvider

            table = Table(box=box.SIMPLE, header_style=None)

            table.add_column("Module")
            table.add_column("Description")

            def is_insight(module: type) -> bool:
                return "insight" in module.__module__

            for modules in (
                RayTracer,
                FeatureExtractor,
                WaveformProvider,
                StationCorrections,
            ):
                table.add_row(f"[bold]{modules.__name__}")
                for module in modules.get_subclasses():
                    name = module.__name__
                    if is_insight(module):
                        name += " 🔑"
                    table.add_row(f" {name}", module.__doc__, style="dim")
                table.add_section()

            console.print(table)

        case "dump-schemas":
            from qseek.models.detection import EventDetections

            if not args.folder.exists():
                raise EnvironmentError(f"folder {args.folder} does not exist")

            file = args.folder / "search.schema.json"
            print(f"writing JSON schemas to {args.folder}")
            file.write_text(json.dumps(Search.model_json_schema(), indent=2))

            file = args.folder / "detections.schema.json"
            file.write_text(json.dumps(EventDetections.model_json_schema(), indent=2))
        case _:
            parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
