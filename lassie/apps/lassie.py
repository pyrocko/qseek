from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from pkg_resources import get_distribution

from lassie.console import console
from lassie.images import ImageFunctions
from lassie.images.phase_net import PhaseNet
from lassie.models import Stations
from lassie.search import SquirrelSearch
from lassie.server import WebServer
from lassie.station_corrections import StationCorrections
from lassie.tracers import RayTracers
from lassie.tracers.cake import CakeTracer
from lassie.utils import ANSI, setup_rich_logging

setup_rich_logging()

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
        "run",
        help="start a new detection run",
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
        "station-corrections",
        help="analyse station corrections from existing run",
        description="analyze and plot station corrections from a finished run",
    )
    station_corrections.add_argument("rundir", type=Path, help="path of existing run")

    serve = subparsers.add_parser(
        "serve",
        help="serve results from an existing run",
        description="start a webserver and serve detections and results from a run",
    )
    serve.add_argument("rundir", type=Path, help="rundir to serve")

    init = subparsers.add_parser(
        "init-project",
        help="initialize a new project",
    )
    init.add_argument("folder", type=Path, help="folder to initialize project in")

    dump_schemas = subparsers.add_parser(
        "dump-schemas",
        help="dump models to json-schema (development)",
    )
    dump_schemas.add_argument("folder", type=Path, help="folder to dump schemas to")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO - args.verbose * 10)

    if args.command == "init-project":
        folder: Path = args.folder
        if folder.exists():
            raise FileExistsError(f"Folder {folder} already exists")
        folder.mkdir()

        pyrocko_stations = folder / "pyrocko-stations.yaml"
        pyrocko_stations.touch()

        config = SquirrelSearch(
            ray_tracers=RayTracers(__root__=[CakeTracer()]),
            image_functions=ImageFunctions(
                __root__=[PhaseNet(phase_map={"P": "cake:P", "S": "cake:S"})]
            ),
            stations=Stations(pyrocko_station_yamls=[pyrocko_stations]),
            waveform_data=[Path("/data/")],
            time_span=(
                datetime.fromisoformat("2023-04-11T00:00:00+00:00"),
                datetime.fromisoformat("2023-04-18T00:00:00+00:00"),
            ),
        )

        config_file = folder / "config.json"
        config_file.write_text(config.json(by_alias=False, indent=2))
        logger.info("initialized new project in folder %s", folder)
        logger.info(
            "start detecting with:\n\t%slassie run config.json%s", ANSI.Bold, ANSI.Reset
        )

    elif args.command == "run":
        search = SquirrelSearch.parse_file(args.config)
        search.init_rundir(force=args.force)

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.scan_squirrel()
            await http

        asyncio.run(_run())

    elif args.command == "continue":
        search = SquirrelSearch.load_rundir(args.rundir)
        search_time_file = args.rundir / "search_progress_time.txt"
        search.search_progress_time = datetime.fromisoformat(
            search_time_file.read_text().strip()
        )

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.scan_squirrel()
            await http

        console.rule("Continuing search")
        asyncio.run(_run())

    elif args.command == "feature-extraction":
        search = SquirrelSearch.load_rundir(args.rundir)

        async def extract() -> None:
            for detection in search._detections.detections:
                await search.add_features(detection)

        asyncio.run(extract())

    elif args.command == "station-corrections":
        search = SquirrelSearch.load_rundir(args.rundir)
        station_corrections = StationCorrections.from_detections(search._detections)
        station_corrections.save_plots(search._rundir / "station-corrections")
        station_corrections.save_csv(
            filename=args.rundir / "station-corrections-stats.csv"
        )

    elif args.command == "serve":
        search = SquirrelSearch.load_rundir(args.rundir)
        webserver = WebServer(search)

        loop = asyncio.get_event_loop()
        loop.create_task(webserver.start())
        loop.run_forever()

    elif args.command == "dump-schemas":
        from lassie.models.detection import Detections

        if not args.folder.exists():
            raise EnvironmentError(f"folder {args.folder} does not exist")

        file = args.folder / "search.schema.json"
        print(f"writing JSON schemas to {args.folder}")
        file.write_text(SquirrelSearch.schema_json(indent=2))

        file = args.folder / "detections.schema.json"
        file.write_text(Detections.schema_json(indent=2))


if __name__ == "__main__":
    main()
