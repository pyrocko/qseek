from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import nest_asyncio
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
    station_corrections.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot station correction results and save to rundir",
    )
    station_corrections.add_argument("rundir", type=Path, help="path of existing run")

    serve = subparsers.add_parser(
        "serve",
        help="serve results from an existing run",
        description="start a webserver and serve detections and results from a run",
    )
    serve.add_argument("rundir", type=Path, help="rundir to serve")

    new = subparsers.add_parser(
        "new",
        help="initialize a new project",
    )
    new.add_argument("folder", type=Path, help="folder to initialize project in")

    dump_schemas = subparsers.add_parser(
        "dump-schemas",
        help="dump models to json-schema (development)",
    )
    dump_schemas.add_argument("folder", type=Path, help="folder to dump schemas to")

    args = parser.parse_args()
    setup_rich_logging(level=logging.INFO - args.verbose * 10)

    if args.command == "new":
        folder: Path = args.folder
        if folder.exists():
            raise FileExistsError(f"Folder {folder} already exists")
        folder.mkdir()

        pyrocko_stations = folder / "pyrocko-stations.yaml"
        pyrocko_stations.touch()

        config = SquirrelSearch(
            ray_tracers=RayTracers(root=[CakeTracer()]),
            image_functions=ImageFunctions(
                root=[PhaseNet(phase_map={"P": "cake:P", "S": "cake:S"})]
            ),
            stations=Stations(pyrocko_station_yamls=[pyrocko_stations]),
            waveform_data=[Path("/data/")],
            time_span=(
                datetime.fromisoformat("2023-04-11T00:00:00+00:00"),
                datetime.fromisoformat("2023-04-18T00:00:00+00:00"),
            ),
        )

        config_file = folder / "config.json"
        config_file.write_text(config.model_dump_json(by_alias=False, indent=2))
        logger.info("initialized new project in folder %s", folder)
        logger.info(
            "start detecting with:\n\t%slassie run config.json%s", ANSI.Bold, ANSI.Reset
        )

    elif args.command == "run":
        search = SquirrelSearch.from_config(args.config)
        search.init_rundir(force=args.force)

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.scan_squirrel()
            await http

        asyncio.run(_run())

    elif args.command == "continue":
        search = SquirrelSearch.load_rundir(args.rundir)
        if search.progress.time_progress:
            console.rule(f"Continuing search from {search.progress.time_progress}")
        else:
            console.rule("Starting search from scratch")

        webserver = WebServer(search)

        async def _run() -> None:
            http = asyncio.create_task(webserver.start())
            await search.scan_squirrel()
            await http

        asyncio.run(_run())

    elif args.command == "feature-extraction":
        search = SquirrelSearch.load_rundir(args.rundir)

        async def extract() -> None:
            for detection in search._detections.detections:
                await search.add_features(detection)

        asyncio.run(extract())

    elif args.command == "station-corrections":
        rundir = Path(args.rundir)
        station_corrections = StationCorrections(rundir=rundir)
        if args.plot:
            station_corrections.save_plots(rundir / "station_corrections")
        station_corrections.save_csv(filename=rundir / "station_corrections_stats.csv")

    elif args.command == "serve":
        search = SquirrelSearch.load_rundir(args.rundir)
        webserver = WebServer(search)

        loop = asyncio.get_event_loop()
        loop.create_task(webserver.start())
        loop.run_forever()

    elif args.command == "dump-schemas":
        from lassie.models.detection import EventDetections

        if not args.folder.exists():
            raise EnvironmentError(f"folder {args.folder} does not exist")

        file = args.folder / "search.schema.json"
        print(f"writing JSON schemas to {args.folder}")
        file.write_text(SquirrelSearch.model_json_schema(indent=2))

        file = args.folder / "detections.schema.json"
        file.write_text(EventDetections.model_json_schema(indent=2))


if __name__ == "__main__":
    main()
