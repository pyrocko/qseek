from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from nicegui import binding

from qseek.models.catalog import EventCatalog
from qseek.search import Search
from qseek.ui.models import CatalogProxy


class RunExplorer(Protocol):
    """Provides an interface to explore runs from local dir / SSH / WebSocket.

    Args:
        source (str): Name of the source, e.g. "local", "ssh", "websocket".
    """

    source: Literal["local", "ssh"]

    async def discover(self) -> AsyncIterator[RunSource]: ...


class RunSource(Protocol):
    """Provides an interface to get run details from local dir / SSH / WebSocket.

    Args:
        source (str): Name of the source, e.g. "local", "ssh", "websocket".
    """

    source: Literal["local", "ssh"]

    name: str

    n_events: int = binding.BindableProperty()
    last_update: datetime = binding.BindableProperty()

    hash: str
    updated: asyncio.Event

    _catalog: CatalogProxy | None = None
    _search: Search | None = None

    async def get_search_json(self) -> Path: ...

    async def get_catalog_path(self) -> Path: ...

    async def get_catalog(self) -> CatalogProxy:
        if not self._catalog:
            catalog_dir = await self.get_catalog_path()
            catalog = await asyncio.to_thread(EventCatalog.load_rundir, catalog_dir)
            self._catalog = CatalogProxy(catalog)
        return self._catalog

    async def get_search(self) -> Search:
        if not self._search:
            search_file = await self.get_search_json()
            self._search = Search.model_validate_json(
                search_file.read_bytes(), context={"assume_validated": True}
            )
        return self._search
