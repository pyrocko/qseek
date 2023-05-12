from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from pkg_resources import get_distribution

from lassie.config import Config
from lassie.models import Stations
from lassie.search import Search


def main():
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

    run = subparsers.add_parser("run", help="start a run")
    run.add_argument(
        "config",
        type=Path,
        help="path to config file",
    )

    subparsers.add_parser("dump-config", help="print a config template to terminal")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO - args.verbose * 10)

    if args.command == "dump-config":
        config = Config.construct(
            stations=Stations.construct(
                pyrocko_station_yamls=[Path("stations.yaml")],
                blacklist=["NE.STA.LOC"],
            ),
            waveform_data=[Path("/data/")],
            time_span=(
                datetime.fromisoformat("2023-04-11T00:00:00+00:00"),
                datetime.fromisoformat("2023-04-18T00:00:00+00:00"),
            ),
        )
        print(config.json(by_alias=False, indent=2))
        return

    elif args.command == "run":
        config = Config.parse_file(args.config)

        search = Search(
            stations=config.stations,
            octree=config.octree,
            ray_tracers=config.ray_tracers,
            image_functions=config.image_functions,
        )

        search.scan_squirrel(
            config.get_squirrel(),
            start_time=config.time_span[0],
            end_time=config.time_span[1],
        )
