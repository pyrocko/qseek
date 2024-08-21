#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import nest_asyncio
from pkg_resources import get_distribution

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection

nest_asyncio.apply()

logger = logging.getLogger(__name__)


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

subparsers.add_parser(
    "config",
    help="print a new config",
    description="Print a new default config configuration file.",
)

search = subparsers.add_parser(
    "search",
    help="start a search",
    description="detect, localize and characterize earthquakes in a dataset",
)
search_config = search.add_argument(
    "config",
    type=Path,
    help="path to config file",
)
search.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="backup old rundir and create a new",
)

continue_run = subparsers.add_parser(
    "continue",
    help="continue an existing search",
    description="Continue a run from an existing rundir",
)
continue_rundir = continue_run.add_argument(
    "rundir",
    type=Path,
    help="existing runding to continue",
)

snuffler = subparsers.add_parser(
    "snuffler",
    help="start the Pyrocko snuffler to inspect waveforms, events and picks",
    description="Start the snuffler to inspect and validate "
    "the detections and waveforms for an existing run",
)
snuffler_rundir = snuffler.add_argument(
    "rundir",
    type=Path,
    help="path of the existing rundir",
)
snuffler.add_argument(
    "--show-picks",
    action="store_true",
    default=False,
    help="load and show picks in snuffler",
)
snuffler.add_argument(
    "--show-semblance",
    action="store_true",
    default=False,
    help="show semblance trace in snuffler",
)


features_extract = subparsers.add_parser(
    "feature-extraction",
    help="extract features from an existing run",
    description="Modify the search.json for re-evaluation of the event's features",
)
features_rundir = features_extract.add_argument(
    "rundir",
    type=Path,
    help="path of existing run",
)
features_extract.add_argument(
    "--recalculate",
    action="store_true",
    default=False,
    help="recalculate all magnitudes",
)
features_extract.add_argument(
    "--nparallel",
    type=int,
    default=32,
    help="number of parallel tasks for feature extraction",
)

modules = subparsers.add_parser(
    "modules",
    help="show available modules",
    description="Show all available modules",
)
modules.add_argument(
    "name",
    nargs="?",
    type=str,
    help="Name of the module to print JSON config for.",
)

serve = subparsers.add_parser(
    "serve",
    help="start webserver and serve results from an existing run",
    description="Start a webserver and serve detections and results from a run",
)
serve.add_argument(
    "rundir",
    type=Path,
    help="rundir to serve",
)


export = subparsers.add_parser(
    "export",
    help="export detections to different output formats",
    description="Export detections to different output formats."
    " Get an overview with `qseek export list`",
)

export.add_argument(
    "format",
    type=str,
    help="Name of export module, or `list` to list available modules",
)

export.add_argument(
    "rundir",
    type=Path,
    help="path to existing qseek rundir",
    nargs="?",
)

export.add_argument(
    "export_dir",
    nargs="?",
    type=Path,
    help="path to export directory",
)

export.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="overwrite existing output directory",
)


subparsers.add_parser(
    "clear-cache",
    help="clear the cach directory",
    description="Clear all data in the cache directory",
)

dump_schemas = subparsers.add_parser(
    "dump-schemas",
    help="dump data models to json-schema (development)",
    description="dDump data models to json-schema, "
    "this is for development purposes only",
)
dump_dir = dump_schemas.add_argument(
    "folder",
    type=Path,
    help="folder to dump schemas to",
)


try:
    import argcomplete
    from argcomplete.completers import DirectoriesCompleter, FilesCompleter

    search_config.completer = FilesCompleter(["*.json"])
    continue_rundir.completer = DirectoriesCompleter()
    snuffler_rundir.completer = DirectoriesCompleter()
    features_rundir.completer = DirectoriesCompleter()
    dump_dir.completer = DirectoriesCompleter()

    argcomplete.autocomplete(parser)
except ImportError:
    pass


def main() -> None:
    from qseek.utils import CACHE_DIR, load_insights, setup_rich_logging

    load_insights()
    from rich import box
    from rich.progress import Progress
    from rich.table import Table

    from qseek.console import console
    from qseek.search import Search
    from qseek.server import WebServer

    args = parser.parse_args()

    log_level = logging.INFO - args.verbose * 10
    loop_debug = log_level < logging.INFO
    setup_rich_logging(level=log_level)

    match args.command:
        case "config":
            config = Search()
            console.print_json(config.model_dump_json(by_alias=False, indent=2))

        case "search":
            search = Search.from_config(args.config)

            webserver = WebServer(search)

            async def run() -> None:
                http = asyncio.create_task(webserver.start())
                await search.start(force_rundir=args.force)
                await http

            asyncio.run(run(), debug=loop_debug)

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

            asyncio.run(run(), debug=loop_debug)

        case "snuffler":
            search = Search.load_rundir(args.rundir)
            squirrel = search.data_provider.get_squirrel()
            show_picks = args.show_picks
            if args.show_semblance:
                squirrel.add([str(search._rundir / "semblance.mseed")])
            squirrel.snuffle(
                events=None if show_picks else search.catalog.as_pyrocko_events(),
                markers=search.catalog.get_pyrocko_markers() if show_picks else None,
                stations=search.stations.as_pyrocko_stations(),
            )

        case "feature-extraction":
            search = Search.load_rundir(args.rundir)
            search.data_provider.prepare(search.stations)
            recalculate_magnitudes = args.recalculate

            tasks = []

            def console_status(task: asyncio.Task[EventDetection]):
                detection = task.result()
                if detection.magnitudes:
                    console.print(
                        f"Event {str(detection.time).split('.')[0]}:",
                        ", ".join(
                            f"[bold]{m.magnitude}[/bold] {m.average:.2f}±{m.error:.2f}"
                            for m in detection.magnitudes
                        ),
                    )
                else:
                    console.print(f"Event {detection.time}: No magnitudes")

            progress = Progress()
            tracker = progress.add_task(
                "Calculating magnitudes",
                total=search.catalog.n_events,
                console=console,
            )

            async def worker() -> None:
                for magnitude in search.magnitudes:
                    await magnitude.prepare(search.octree, search.stations)
                await search.catalog.check(repair=True)

                sem = asyncio.Semaphore(args.nparallel)
                for detection in search.catalog:
                    await sem.acquire()
                    task = asyncio.create_task(
                        search.add_magnitude_and_features(
                            detection,
                            recalculate=recalculate_magnitudes,
                        )
                    )
                    tasks.append(task)
                    task.add_done_callback(lambda _: sem.release())
                    task.add_done_callback(tasks.remove)
                    task.add_done_callback(console_status)
                    task.add_done_callback(
                        lambda _: progress.update(tracker, advance=1)
                    )

                await asyncio.gather(*tasks)

                await search._catalog.save()
                await search._catalog.export_detections(
                    jitter_location=search.octree.smallest_node_size()
                )

            asyncio.run(worker(), debug=loop_debug)

        case "serve":
            search = Search.load_rundir(args.rundir)
            webserver = WebServer(search)

            async def start() -> None:
                await webserver.start()

            asyncio.run(start(), debug=loop_debug)

        case "clear-cache":
            logger.info("clearing cache directory %s", CACHE_DIR)
            shutil.rmtree(CACHE_DIR)

        case "export":
            from qseek.exporters.base import Exporter

            def show_table():
                table = Table(box=box.SIMPLE, header_style=None)
                table.add_column("Exporter")
                table.add_column("Description")
                for exporter in Exporter.get_subclasses():
                    table.add_row(
                        f"[bold]{exporter.__name__.lower()}",
                        exporter.__doc__,
                    )
                console.print(table)

            if args.format == "list":
                show_table()
                parser.exit()

            if not args.rundir:
                parser.error("rundir is required for export")

            if args.export_dir is None:
                parser.error("export directory is required")

            if args.export_dir.exists():
                if not args.force:
                    parser.error(f"export directory {args.export_dir} already exists")
                shutil.rmtree(args.export_dir)

            for exporter in Exporter.get_subclasses():
                if exporter.__name__.lower() == args.format.lower():
                    exporter_instance = exporter()
                    asyncio.run(
                        exporter_instance.export(
                            rundir=args.rundir,
                            outdir=args.export_dir,
                        )
                    )
                    break
            else:
                available_exporters = ", ".join(
                    exporter.__name__ for exporter in Exporter.get_subclasses()
                )
                parser.error(
                    f"unknown exporter: {args.format}"
                    f"choose fom: {available_exporters}"
                )

        case "modules":
            from qseek.corrections.base import TravelTimeCorrections
            from qseek.features.base import FeatureExtractor
            from qseek.magnitudes.base import EventMagnitudeCalculator
            from qseek.pre_processing.base import BatchPreProcessing
            from qseek.tracers.base import RayTracer
            from qseek.waveforms.base import WaveformProvider

            table = Table(box=box.SIMPLE, header_style=None)

            table.add_column("Module")
            table.add_column("Description")

            module_classes = (
                WaveformProvider,
                BatchPreProcessing,
                RayTracer,
                FeatureExtractor,
                EventMagnitudeCalculator,
                TravelTimeCorrections,
            )

            if args.name:
                for module in module_classes:
                    for subclass in module.get_subclasses():
                        if subclass.__name__ == args.name:
                            console.print_json(subclass().model_dump_json(indent=2))
                            parser.exit()
                else:
                    parser.error(f"unknown module: {args.name}")

            def is_insight(module: type) -> bool:
                return "insight" in module.__module__

            for modules in module_classes:
                table.add_row(f"[bold]{modules.__name__}")
                for module in modules.get_subclasses():
                    name = module.__name__
                    if is_insight(module):
                        name += " 🚀"
                    table.add_row(f" {name}", module.__doc__, style="dim")
                table.add_section()

            console.print(table)
            console.print("Insight module are marked by 🚀\n")
            console.print(
                "Use [bold]qseek modules <module_name>[/bold] "
                "to print the JSON schema"
            )

        case "dump-schemas":
            import json

            from qseek.models.catalog import EventCatalog

            if not args.folder.exists():
                raise EnvironmentError(f"folder {args.folder} does not exist")

            file = args.folder / "search.schema.json"
            console.print(f"writing JSON schemas to {args.folder}")
            file.write_text(json.dumps(Search.model_json_schema(), indent=2))

            file = args.folder / "detections.schema.json"
            file.write_text(json.dumps(EventCatalog.model_json_schema(), indent=2))
        case _:
            parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
