import asyncio
import logging
from collections.abc import AsyncIterator
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import aiofiles
import aiohttp
import rfc3986

from qseek.ui.explorer.base import RunExplorer, RunSource
from qseek.utils import datetime_now

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
