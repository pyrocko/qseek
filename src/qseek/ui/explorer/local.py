from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import awatch

from qseek.models.detection import FILENAME_DETECTIONS, FILENAME_RECEIVERS
from qseek.ui.explorer.base import RunExplorer, RunSource

if TYPE_CHECKING:
    from qseek.ui.state import CatalogStore

logger = logging.getLogger(__name__)


def _count_lines(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    with path.open() as f:
        return sum(1 for _ in f)


class LocalSource(RunSource):
    source = "local"

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

        search_json = self.run_dir / "search.json"
        if not search_json.is_file():
            raise ValueError(f"search.json not found in {self.run_dir}")
        detections_json = self.run_dir / FILENAME_DETECTIONS
        detections_receivers = self.run_dir / FILENAME_RECEIVERS

        if not detections_json.is_file() or not detections_receivers.is_file():
            raise ValueError(
                f"{FILENAME_DETECTIONS} or {FILENAME_RECEIVERS} "
                f"not found in {self.run_dir}"
            )
        self.name = self.run_dir.name
        self.hash = hashlib.sha1(search_json.read_bytes()).hexdigest()

        self.last_update = datetime.fromtimestamp(search_json.stat().st_ctime)  # noqa
        self.n_events = _count_lines(detections_json)
        self._task = asyncio.create_task(self.watch_for_updates())

        self.updated = asyncio.Condition()

    async def get_search_json(self) -> Path:
        return self.run_dir / "search.json"

    async def attach(self, proxy: CatalogStore) -> None:
        """Local runs notify their updates with the `updated` condition."""

    async def detach(self, proxy: CatalogStore) -> None:
        """Local runs notify their updates with the `updated` condition."""

    async def get_catalog_path(self) -> Path:
        return self.run_dir

    async def watch_for_updates(self):
        detections_json = self.run_dir / FILENAME_DETECTIONS
        async for changes in awatch(detections_json):
            for _ in changes:
                logger.info("Detected change in %s", detections_json)
                self.n_events = _count_lines(detections_json)
                async with self.updated:
                    self.updated.notify_all()


class LocalRunExplorer(RunExplorer):
    source = "local"

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    async def discover(self) -> AsyncIterator[RunSource]:
        rundirs = sorted(
            list(self.runs_dir.glob("*/search.json"))
            + list(self.runs_dir.glob("search.json")),
            key=lambda p: p.stat().st_ctime,
            reverse=True,
        )
        for search_json in rundirs:
            run_dir = search_json.parent
            logger.info("Found run at %s", run_dir)
            try:
                yield LocalSource(run_dir)
            except ValueError as e:
                logger.warning("Skipping run at %s: %s", run_dir, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LocalRunDiscoverer")
    parser.add_argument("runs_dir", type=Path, help="Directory containing runs")
    args = parser.parse_args()
    discoverer = LocalRunExplorer(args.runs_dir)

    async def main():
        async for run in discoverer.discover():
            print(  # noqa
                f"Found run created at {run.last_update} with {run.n_events} events"
            )

    asyncio.run(main())
