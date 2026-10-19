#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import asyncio
import logging
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

LOG_FILE = "inversion.log"
CONFIG_FILE = "inversion.json"

parser = argparse.ArgumentParser(
    prog="qseek-inversion",
    description="Data-driven velocity model inversion from a qseek run",
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
    "--non-interactive",
    action="store_true",
    default=False,
    help="log only errors to the console, plus a few `key: value` lines with the "
    "rundir, the log file and the result of the run. For scripts and agents; the "
    f"full log is in the `{LOG_FILE}` of the rundir",
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

config = subparsers.add_parser(
    "config",
    help="print a new config",
    description="Print a new default inversion config.",
)
config_import_rundir = config.add_argument(
    "--import-rundir",
    type=Path,
    default=None,
    help="qseek run directory to invert the velocity model for",
)
config.add_argument(
    "--selection",
    type=str,
    default=None,
    help="event and waveform selection module, see `qseek-inversion modules`",
)

invert = subparsers.add_parser(
    "invert",
    help="start an inversion",
    description="Invert for the velocity model that focuses the image functions "
    "of the selected events best",
)
invert_config = invert.add_argument(
    "config",
    type=Path,
    help="path to the inversion config file",
)
invert_rundir = invert.add_argument(
    "--rundir",
    type=Path,
    default=None,
    help="directory of the inversion run. "
    "Default is the config path without its suffix",
)
invert.add_argument(
    "--check",
    action="store_true",
    default=False,
    help="validate the config and select the events, "
    "then exit without creating a rundir",
)
invert.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="move an existing rundir to a backup and create a new one",
)

modules = subparsers.add_parser(
    "modules",
    help="show available modules",
    description="Show the inversions and the event and waveform selections",
)
modules.add_argument(
    "name",
    nargs="?",
    type=str,
    help="name of the module to print the JSON config for",
)

summary = subparsers.add_parser(
    "summary",
    help="print the status of an inversion run as key: value lines",
    description="Print the paths and the result of an existing inversion run",
)
summary_rundir = summary.add_argument(
    "rundir",
    type=Path,
    help="path of an existing inversion run",
)

try:
    import argcomplete
    from argcomplete.completers import DirectoriesCompleter, FilesCompleter

    completers = {
        config_import_rundir: DirectoriesCompleter(),
        invert_config: FilesCompleter(["*.json"]),
        invert_rundir: DirectoriesCompleter(),
        summary_rundir: DirectoriesCompleter(),
    }
    for action, completer in completers.items():
        # argcomplete reads the completer attribute set on the actions
        cast("CompletedAction", action).completer = completer

    argcomplete.autocomplete(parser)
except ImportError:
    pass


class CompletedAction(Protocol):
    """An argparse action with an argcomplete completer."""

    completer: Any


class UsageError(Exception):
    """A wrong path or argument given by the user."""


EXIT_CONFIG = 2
EXIT_ERROR = 1
EXIT_INTERRUPT = 130


def get_modules() -> dict[str, list[type[BaseModel]]]:
    """The modules of an inversion config, by kind."""
    from qseek_inversion.inversion import InversionLayered1D
    from qseek_inversion.waveforms import WaveformSelection

    return {
        "Inversion": [InversionLayered1D],
        "WaveformSelection": sorted(
            WaveformSelection.get_subclasses(), key=lambda m: m.__name__
        ),
    }


def get_module(name: str) -> type[BaseModel]:
    for module_classes in get_modules().values():
        for module in module_classes:
            if module.__name__.lower() == name.lower():
                return module
    available = ", ".join(
        module.__name__ for modules in get_modules().values() for module in modules
    )
    raise UsageError(f"unknown module {name}, choose from: {available}")


def get_rundir(config_file: Path, rundir: Path | None) -> Path:
    return rundir or config_file.with_suffix("")


def backup_rundir(rundir: Path) -> Path:
    from qseek.utils import datetime_now

    backup = rundir.with_name(
        f"{rundir.name}.bak-{datetime_now():%Y%m%dT%H%M%S}",
    )
    rundir.rename(backup)
    logger.info("moved existing rundir to %s", backup)
    return backup


def summarize_rundir(rundir: Path) -> None:
    from qseek.console import report
    from qseek_inversion.inversion import InversionSummary

    if not (rundir / CONFIG_FILE).is_file():
        raise UsageError(f"{rundir} is not a qseek-inversion rundir")
    report("rundir", rundir.resolve())
    report("log", (rundir / LOG_FILE).resolve())
    report("config", (rundir / CONFIG_FILE).resolve())
    iterations_file = rundir / "iterations.jsonl"
    if iterations_file.is_file():
        n_lines = sum(1 for _ in iterations_file.open())
        report("iterations", iterations_file.resolve())
        report("trials", max(n_lines - 1, 0))
    results_file = rundir / "results.json"
    if results_file.is_file():
        result = InversionSummary.model_validate_json(results_file.read_text())
        report("results", results_file.resolve())
        report("best_model", (rundir / "best_model.json").resolve())
        report("start_semblance", f"{result.start_semblance:.4f}")
        report("best_semblance", f"{result.best_semblance:.4f}")
        report("best_iteration", result.best_iteration)
    report("status", "finished" if results_file.is_file() else "incomplete")


def main() -> None:
    from pydantic import ValidationError

    try:
        run()
    except KeyboardInterrupt:
        _report_failure("interrupted", EXIT_INTERRUPT)
    except (UsageError, FileExistsError, ValidationError) as exc:
        logger.error("%s", exc)
        _report_failure(f"{type(exc).__name__}: {exc}".splitlines()[0], EXIT_CONFIG)
    except Exception as exc:
        logger.exception("qseek-inversion failed")
        _report_failure(f"{type(exc).__name__}: {exc}".splitlines()[0], EXIT_ERROR)


def _report_failure(error: str, code: int) -> None:
    from qseek.console import report

    report("status", "failed")
    report("error", error)
    raise SystemExit(code)


def run() -> None:
    from qseek import console as console_module
    from qseek.console import console, report
    from qseek.utils import setup_rich_logging

    args = parser.parse_args()

    if args.non_interactive:
        console.quiet = True
        console_module.NON_INTERACTIVE = True
        log_level = logging.ERROR - args.verbose * 10
    else:
        log_level = logging.INFO - args.verbose * 10
    if args.command == "invert":
        report("qseek-inversion", f"{version('qseek')} {args.command}")
    # The run's log file keeps INFO messages when the console is quiet
    setup_rich_logging(level=log_level, file_level=logging.INFO)

    match args.command:
        case "config":
            logging.disable(logging.CRITICAL)
            from qseek_inversion.inversion import InversionLayered1D

            inversion = InversionLayered1D()
            if args.import_rundir:
                inversion.import_rundir = args.import_rundir.resolve()
            if args.selection:
                from qseek_inversion.waveforms import WaveformSelection

                selection = get_module(args.selection)
                if not issubclass(selection, WaveformSelection):
                    raise UsageError(f"{args.selection} is not a waveform selection")
                inversion.event_selection = selection.model_validate({})
            console.quiet = False
            console.print_json(inversion.model_dump_json(indent=2))

        case "modules":
            from rich import box
            from rich.table import Table

            if args.name:
                logging.disable(logging.CRITICAL)
                module = get_module(args.name)
                console.quiet = False
                console.print_json(module.model_validate({}).model_dump_json(indent=2))
                return

            table = Table(box=box.SIMPLE, header_style=None)
            table.add_column("Module")
            table.add_column("Description")
            for kind, module_classes in get_modules().items():
                table.add_row(f"[bold]{kind}")
                for module in module_classes:
                    module_doc = (module.__doc__ or "").strip().splitlines() or [""]
                    table.add_row(f" {module.__name__}", module_doc[0], style="dim")
                table.add_section()
            console.print(table)
            console.print(
                "Use [bold]qseek-inversion modules <module_name>[/bold] "
                "to print the JSON config"
            )

        case "invert":
            from qseek_inversion.inversion import InversionLayered1D

            if not args.config.is_file():
                raise UsageError(f"config {args.config} does not exist")
            inversion = InversionLayered1D.model_validate_json(args.config.read_text())

            if args.check:
                console_module.NON_INTERACTIVE = True
                asyncio.run(inversion.prepare())
                stations = inversion.event_selection.get_stations()
                report("import_rundir", inversion.import_rundir.resolve())
                report("stations", stations.n_stations)
                report("events", inversion.event_selection.n_selected_events())
                report("status", "ok")
                return

            rundir = get_rundir(args.config, args.rundir)
            if rundir.exists():
                if not args.force:
                    raise FileExistsError(
                        f"rundir {rundir} exists, use --force to move it to a backup"
                    )
                backup_rundir(rundir)
            rundir.mkdir(parents=True)

            file_logger = logging.FileHandler(rundir / LOG_FILE)
            file_logger.setLevel(logging.INFO)
            logging.root.addHandler(file_logger)
            # basicConfig leaves the level alone when the root has handlers already
            logging.root.setLevel(min(logging.root.level, logging.INFO))

            (rundir / CONFIG_FILE).write_text(inversion.model_dump_json(indent=2))
            report("rundir", rundir.resolve())
            report("log", (rundir / LOG_FILE).resolve())

            result = asyncio.run(inversion.invert(rundir))

            report("results", (rundir / "results.json").resolve())
            report("best_model", (rundir / "best_model.json").resolve())
            report("iterations", (rundir / "iterations.jsonl").resolve())
            report("events", result.n_selected_events)
            report("start_semblance", f"{result.start_semblance:.4f}")
            report("best_semblance", f"{result.best_semblance:.4f}")
            report("best_iteration", result.best_iteration)
            report("duration", str(result.duration).split(".")[0])
            report("status", "finished")

        case "summary":
            console_module.NON_INTERACTIVE = True
            summarize_rundir(args.rundir)

        case _:
            parser.error(f"unknown command: {args.command}")


if __name__ in {"__main__", "__mp_main__"}:
    main()
