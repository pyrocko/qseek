from __future__ import annotations

import logging
from pathlib import Path

from nicegui import app

from qseek.ui.explorer import LocalRunExplorer, SshExplorer
from qseek.ui.explorer.base import RunSource
from qseek.ui.state import get_tab_state

logger = logging.getLogger(__name__)


class SourceManager:
    runs: dict[str, RunSource]

    def __init__(self):
        self.explorers = []
        self.runs = {}

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

    async def set_active_run(self, hash: str) -> None:
        if hash not in self.runs:
            raise ValueError(f"Run with hash {hash} not found")
        old_hash = app.storage.tab.get("active_run")
        app.storage.tab["active_run"] = hash
        if old_hash != hash:
            logger.info("Active run changed to %s", hash)
            tab_state = get_tab_state()
            await tab_state.set_run(self.runs[hash])
