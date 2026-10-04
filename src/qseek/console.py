from rich.console import Console

console = Console()

NON_INTERACTIVE = False
"""Set by `qseek --non-interactive`: no rich output, only `report` lines."""


def report(key: str, value: object) -> None:
    """Print one `key: value` line to stdout in non-interactive mode.

    These lines are the whole console output of a non-interactive run, besides
    errors, and meant to be read by scripts and agents.
    """
    if NON_INTERACTIVE:
        print(f"{key}: {value}", flush=True)  # noqa: T201
