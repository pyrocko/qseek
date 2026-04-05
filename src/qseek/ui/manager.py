from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

from nicegui import app

from qseek.ui.explorer import LocalRunExplorer, SshExplorer
from qseek.ui.explorer.base import RunSource

logger = logging.getLogger(__name__)


class SourceManager:
    runs: dict[str, RunSource]
    _callbacks: list[Callable[[], None]]

    def __init__(self):
        self.explorers = []
        self.runs = {}
        self._callbacks = []

    async def add_uri(self, uri: str):
        uri_path = Path(uri)
        if uri_path.exists():
            explorer = LocalRunExplorer(uri_path)
        elif uri.startswith("ssh://"):
            explorer = SshExplorer(uri)
        else:
            raise ValueError(f"Invalid URI: {uri}")
        async for run_source in explorer.discover():
            self.runs[run_source.hash] = run_source
        self.explorers.append(explorer)

    async def add_uris(self, uris: list[str]):
        for uri in uris:
            await self.add_uri(uri)

    def get_run(self, hash: str) -> RunSource:
        return self.runs[hash]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def set_active_run(self, hash: str) -> None:
        if hash not in self.runs:
            raise ValueError(f"Run with hash {hash} not found")
        old_hash = app.storage.tab.get("active_run")
        app.storage.tab["active_run"] = hash
        if old_hash != hash:
            logger.info("Active run changed to %s", hash)
            for callback in self._callbacks:
                callback()

    def get_active_run(self) -> RunSource:
        if self.n_runs == 0:
            raise RuntimeError("No runs loaded")
        try:
            active_run = app.storage.tab["active_run"]
        except (KeyError, RuntimeError):
            active_run = next(iter(self.runs.keys()))
            with contextlib.suppress(ValueError, RuntimeError):
                self.set_active_run(active_run)
        return self.get_run(active_run)

    def on_active_run_change(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)
