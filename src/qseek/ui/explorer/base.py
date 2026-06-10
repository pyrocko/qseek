from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from nicegui import binding

from qseek.models.catalog import EventCatalog
from qseek.search import Search
from qseek.types import allow_non_existing_paths

if TYPE_CHECKING:
    from qseek.ui.state import CatalogStore


logger = logging.getLogger(__name__)


class RunExplorer(Protocol):
    """Provides an interface to explore runs from local dir / SSH / WebSocket.

    Args:
        source (str): Name of the source, e.g. "local", "ssh", "qseek-http".
    """

    source: Literal["local", "ssh", "qseek-http"]

    async def discover(self) -> AsyncIterator[RunSource]: ...


class RunSource(Protocol):
    """Provides an interface to get run details from local dir / SSH / WebSocket.

    Args:
        source (str): Name of the source, e.g. "local", "ssh", "qseek-http".
    """

    source: Literal["local", "ssh", "qseek-http"]

    name: str

    n_events: int = binding.BindableProperty()
    last_update: datetime = binding.BindableProperty()

    hash: str
    updated: asyncio.Condition
    tags: list[str] = []

    live: bool = False

    _catalog: EventCatalog | None = None
    _search: Search | None = None

    async def get_search_json(self) -> Path:
        """Get the path to the search JSON file.

        Returns:
            Path: The path to the search JSON file.
        """

    async def get_catalog_path(self) -> Path:
        """Get the path to the catalog directory.

        Holding detections.json and detections_receivers.json.

        Returns:
            Path: The path to the catalog directory.
        """

    async def get_catalog(self) -> EventCatalog:
        """Get the catalog for the run.

        Returns:
            EventCatalog: The catalog for the run.
        """
        if not self._catalog:
            catalog_dir = await self.get_catalog_path()
            self._catalog = await asyncio.to_thread(
                EventCatalog.load_rundir, catalog_dir
            )
            logger.info(
                "Loaded catalog with %d events from %s",
                self._catalog.n_events,
                self.name,
            )
        return self._catalog

    async def get_search(self) -> Search:
        """Get the search object for the run.

        Returns:
            Search: The search object for the run.
        """
        allow_non_existing_paths(True)
        if not self._search:
            search_file = await self.get_search_json()
            logger.info("Loading search from %s", search_file)
            self._search = await asyncio.to_thread(
                Search.model_validate_json, search_file.read_bytes()
            )
        return self._search

    async def attach(self, proxy: CatalogStore):
        """Attach to the run source for updates.

        This method should be called to start listening for updates from the run source.
        """

    async def detach(self, proxy: CatalogStore):
        """Detach from the run source for updates.

        This method should be called to stop listening for updates from the run source.
        """
