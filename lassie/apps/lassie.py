from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from pkg_resources import get_distribution

from lassie.models import Stations
from lassie.search import SquirrelSearch
from lassie.server import WebServer
from lassie.tracers import CakeTracer, RayTracers

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
    )
    run.add_argument("config", type=Path, help="path to config file")
    run.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="backup old rundir and create a new",
    )

    features = subparsers.add_parser(
        "feature-extraction",
        help="extract features from an existing run",
        description="modify the search.json for re-evaluation of the event's features",
    )
    features.add_argument("rundir", type=Path, help="path of existing run")

    serve = subparsers.add_parser(
        "serve",
        help="serve results from an existing run",
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

        folder.mkdir(exist_ok=True)

        config = SquirrelSearch.construct(
            stations=Stations.construct(
                pyrocko_station_yamls=[Path("pyrocko-stations.yaml")],
                blacklist=["NE.STA.LOC"],
            ),
            waveform_data=[Path("/data/")],
            ray_tracers=RayTracers(__root__=[CakeTracer()]),
            time_span=(
                datetime.fromisoformat("2023-04-11T00:00:00+00:00"),
                datetime.fromisoformat("2023-04-18T00:00:00+00:00"),
            ),
        )
        config_file = folder / "config.json"
        config_file.write_text(config.json(by_alias=False, indent=2))
        (folder / "pyrocko-stations.yaml").touch()
        logger.info("initialized project in folder %s", folder)

    elif args.command == "run":
        search = SquirrelSearch.parse_file(args.config)
        search.init_rundir(force=args.force)

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
