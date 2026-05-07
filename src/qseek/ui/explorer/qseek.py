from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Self

import aiofiles
import aiohttp
import rfc3986
from pydantic import ValidationError

from qseek.server import WebsocketMessage
from qseek.ui.explorer.base import RunExplorer, RunSource
from qseek.utils import datetime_now

if TYPE_CHECKING:
    from qseek.ui.state import ProxyCatalog

logger = logging.getLogger(__name__)


class QseekWebSource(RunSource):
    source = "qseek-http"

    def __init__(
        self,
        remote: str,
        tmp_dir: TemporaryDirectory,
        hash: str,
    ):
        self.name = remote
        self._tmp_dir = tmp_dir
        self._tmp_path = Path(tmp_dir.name)
        self.hash = hash
        self.last_update = datetime_now()
        self.tags = ["live"]
        self.n_events = 0

        self._attached_proxies = set()
        self._websocket_task = None
        self.updated = asyncio.Event()

    async def get_search_json(self) -> Path:
        async with (
            aiohttp.ClientSession(
                base_url=f"{self.name}/api/v1/",
                raise_for_status=True,
            ) as session,
            session.get("search") as resp,
        ):
            search_json = await resp.text()
            search_path = self._tmp_path / "search.json"
            search_path.write_text(search_json)
            return search_path

    async def get_catalog_path(self) -> Path:
        await self._download_catalog_data()
        return self._tmp_path

    async def _listen_websocket(self):
        while True:
            async with (
                aiohttp.ClientSession(
                    base_url=f"{self.name}/api/v1/",
                    raise_for_status=True,
                ) as session,
                session.ws_connect("ws") as ws,
            ):
                logger.info(
                    "listening for WebSocket messages from %s", session._base_url
                )
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            msg = WebsocketMessage.model_validate_json(msg.data)
                            await self._new_websocket_message(msg)
                        except ValidationError as exc:
                            logger.error(
                                "Failed to parse WebSocket message from %s: %s",
                                self.name,
                                exc,
                            )
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(
                            "WebSocket error from %s: %s",
                            self.name,
                            msg.data,
                        )
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info(
                            "WebSocket connection to %s closed by server",
                            self.name,
                        )
                        break

            logger.warning(
                "WebSocket connection to %s closed, reconnecting...",
                self.name,
            )
            await asyncio.sleep(5)  # Wait before trying to reconnect

    async def _new_websocket_message(self, msg: WebsocketMessage):
        logger.debug("Received WebSocket message")
        if msg.type == "NewDetections":
            collection = msg.data
            logger.info(
                "Received %d new detections from %s",
                collection.n_detections(),
                self.name,
            )

            self.n_events += collection.n_detections()
            self.last_update = datetime_now()
            if self._catalog:
                for detection in collection.detections:
                    self._catalog.add(detection)
            self.updated.set()
            self.updated.clear()

    async def _download_catalog_data(self) -> Path:
        logger.info("Fetching detections and receivers from %s", self.name)
        async with aiohttp.ClientSession(
            base_url=f"{self.name}/api/v1/",
            raise_for_status=True,
        ) as session:
            detections_path = self._tmp_path / "detections.json"
            detections_receivers_path = self._tmp_path / "detections_receivers.json"

            logger.debug("Fetching detections from %s", self.name)
            async with (
                session.get("detections") as resp,
                aiofiles.open(detections_path, "wb") as f,
            ):
                async for chunk, _ in resp.content.iter_chunks():
                    await f.write(chunk)
                await f.flush()
            logger.debug("Fetching detection receivers from %s", self.name)
            async with (
                session.get("detections_receivers") as resp,
                aiofiles.open(detections_receivers_path, "wb") as f,
            ):
                async for chunk, _ in resp.content.iter_chunks():
                    await f.write(chunk)
                await f.flush()

            print(self._tmp_path)  # noqa

        return self._tmp_path

    @classmethod
    async def from_remote(cls, remote: str) -> Self:
        tmp_dir = TemporaryDirectory(prefix="qseek-http-")

        async with (
            aiohttp.ClientSession(
                base_url=f"{remote}/api/v1/",
                raise_for_status=True,
            ) as session,
            session.get("search") as resp,
        ):
            resp = await resp.text()
            hash = sha1(resp.encode()).hexdigest()

        return cls(remote, tmp_dir, hash=hash)

    async def attach(self, proxy: ProxyCatalog):
        self._attached_proxies.add(proxy)
        if self._websocket_task is None:
            self._websocket_task = asyncio.create_task(self._listen_websocket())

    async def detach(self, proxy: ProxyCatalog):
        self._attached_proxies.discard(proxy)
        if not self._attached_proxies and self._websocket_task is not None:
            self._websocket_task.cancel()

    def __del__(self):
        self._tmp_dir.cleanup()


class QseekHttpExplorer(RunExplorer):
    def __init__(self, remote_uri: str):
        """Initialize the Qseek HTTP explorer.

        Args:
            remote_uri (str): The base URI of the Qseek HTTP server,
                e.g. "qseek://localhost:19000".
        """
        uri = rfc3986.urlparse(remote_uri)
        if uri.scheme != "qseek":
            raise ValueError(f"Invalid URI scheme: {uri.scheme}, expected 'qseek'")
        self.remote_uri = f"http://{uri.host}:{uri.port or 19000}"

    async def discover(self) -> AsyncIterator[RunSource]:
        try:
            source = await QseekWebSource.from_remote(self.remote_uri)
            yield source
        except Exception as e:
            logger.error("Failed to discover run from %s: %s", self.remote_uri, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test QseekWebSource")
    parser.add_argument("remote", type=str, help="Remote URL of the Qseek HTTP server")
    args = parser.parse_args()

    async def run():
        source = await QseekWebSource.from_remote(args.remote)
        print(f"Run name: {source.name}")  # noqa
        print(f"Hash: {source.hash}")  # noqa
        print(f"Last update: {source.last_update}")  # noqa

        search = await source.get_search()
        print(f"Got the search {type(search)}")  # noqa

        catalog = await source.get_catalog()
        print(f"Number of events: {catalog.n_events}")  # noqa

    asyncio.run(run())
