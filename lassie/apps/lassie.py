from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from pathlib import Path

import nest_asyncio
from pkg_resources import get_distribution

from lassie.console import console
from lassie.models import Stations
from lassie.search import Search
from lassie.server import WebServer
from lassie.station_corrections import StationCorrections
from lassie.utils import CACHE_DIR, setup_rich_logging

nest_asyncio.apply()

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lassie",
        description="The friendly earthquake detector - V2",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="increase verbosity of the log messages, default level is INFO",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_distribution("lassie").version,
        help="show version and exit",
    )

    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")

    run = subparsers.add_parser(
        "search",
        help="start a search 🐕",
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

    init_project = subparsers.add_parser(
        "init",
        help="initialize a new Lassie project",
    )
    init_project.add_argument(
        "folder", type=Path, help="folder to initialize project in"
    )

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

    serve = subparsers.add_parser(
        "serve",
        help="start webserver and serve results from an existing run",
        description="start a webserver and serve detections and results from a run",
    )
    serve.add_argument("rundir", type=Path, help="rundir to serve")

    subparsers.add_parser(
        "clear-cache",
        help="clear the cach directory",
    )

    dump_schemas = subparsers.add_parser(
        "dump-schemas",
        help="dump data models to json-schema (development)",
    )
    dump_schemas.add_argument("folder", type=Path, help="folder to dump schemas to")

    args = parser.parse_args()
    setup_rich_logging(level=logging.INFO - args.verbose * 10)

    if args.command == "init":
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
        logger.info("start detection with: lassie run %s", config_file.name)

    elif args.command == "run":
        search = Search.from_config(args.config)

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.start(force_rundir=args.force)
            await http

        asyncio.run(_run())

    elif args.command == "continue":
        search = Search.load_rundir(args.rundir)
        if search._progress.time_progress:
            console.rule(f"Continuing search from {search._progress.time_progress}")
        else:
            console.rule("Starting search from scratch")

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.start()
            await http

        asyncio.run(_run())

    elif args.command == "feature-extraction":
        search = Search.load_rundir(args.rundir)

        async def extract() -> None:
            for detection in search._detections.detections:
                await search.add_features(detection)

        asyncio.run(extract())

    elif args.command == "corrections":
        rundir = Path(args.rundir)
        station_corrections = StationCorrections(rundir=rundir)
        if args.plot:
            station_corrections.save_plots(rundir / "station_corrections")
        station_corrections.save_csv(filename=rundir / "station_corrections_stats.csv")

    elif args.command == "serve":
        search = Search.load_rundir(args.rundir)
        webserver = WebServer(search)

        loop = asyncio.get_event_loop()
        loop.create_task(webserver.start())
        loop.run_forever()

    elif args.command == "clear-cache":
        logger.info("clearing cache directory %s", CACHE_DIR)
        shutil.rmtree(CACHE_DIR)

    elif args.command == "dump-schemas":
        from lassie.models.detection import EventDetections

        if not args.folder.exists():
            raise EnvironmentError(f"folder {args.folder} does not exist")

        file = args.folder / "search.schema.json"
        print(f"writing JSON schemas to {args.folder}")
        file.write_text(Search.model_json_schema(indent=2))

        file = args.folder / "detections.schema.json"
        file.write_text(EventDetections.model_json_schema(indent=2))


if __name__ == "__main__":
    main()
