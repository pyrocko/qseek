#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import logging
from importlib.metadata import version

import nest_asyncio

from qseek.utils import setup_rich_logging

nest_asyncio.apply()


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    prog="qseek-inversion",
    description="Data-driven velocity model inversion",
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
    version=version("qseek"),
    help="show version and exit",
)
subparsers = parser.add_subparsers(
    title="commands",
    required=True,
    dest="command",
    description="Available commands to run qseek-inversion. Get command help with "
    "`qseek-inversion <command> --help`.",
)
subparsers.add_parser(
    "config",
    help="print a new config",
    description="Print a new default config configuration file.",
)


def main() -> None:
    args = parser.parse_args()

    log_level = logging.INFO - args.verbose * 10
    setup_rich_logging(level=log_level)

    match args.command:
        case "config":
            ...
        case _:
            parser.print_help()
