import argparse
from pathlib import Path

from lassie.config import Config


def main():
    parser = argparse.ArgumentParser(
        prog="lassie",
        description="The friendly earthquake detector - V2",
    )
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")
    subparsers.add_parser("dump-config", help="Print a new config to terminal.")

    run = subparsers.add_parser("run", help="Start a run.")
    run.add_argument(
        "config",
        type=Path,
        help="Path to config file.",
    )

    args = parser.parse_args()
    if args.command == "dump-config":
        config = Config.construct(
            stations_file=Path("stations.yaml"),
            waveform_data=[Path("data/")],
            station_blacklist=["NE.STA.LOC"],
        )
        print(config.json(by_alias=False, indent=2))
        return

    elif args.command == "run":
        ...
    else:
        raise ValueError("No command provided")
