from __future__ import annotations

import logging
from pathlib import Path

import rich
from rich.prompt import FloatPrompt

from qseek.exporters.base import Exporter
from qseek.search import Search

logger = logging.getLogger(__name__)


class Velest(Exporter):
    """Crate a VELEST project folder for 1D velocity model estimation."""

    min_pick_semblance: float = 0.3
    n_picks: dict[str, int] = {}
    n_events: int = 0

    async def export(self, rundir: Path, outdir: Path) -> Path:
        rich.print("Exporting qseek search to VELEST project folder")
        min_pick_semblance = FloatPrompt.ask("Minimum pick confidence", default=0.3)

        self.min_pick_semblance = min_pick_semblance

        outdir.mkdir()
        search = Search.load_rundir(rundir)
        catalog = search.catalog  # noqa

        export_info = outdir / "export_info.json"
        export_info.write_text(self.model_dump_json(indent=2))

        return outdir
