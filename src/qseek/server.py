from __future__ import annotations

import logging
from importlib.metadata import version
from ipaddress import IPv4Address
from typing import TYPE_CHECKING, Literal
from weakref import WeakSet

import aiofiles
import aiohttp_cors
from aiohttp import web
from pydantic import Field, PrivateAttr

from qseek.base import Model

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.search import Search

logger = logging.getLogger(__name__)

MAX_TRIES = 5


class WebServer(Model):
    host: IPv4Address = Field(
        default=IPv4Address("0.0.0.0"),
        description="Host to bind the server to. Default is all interfaces.",
    )
    port: int | Literal["auto"] = Field(
        default="auto",
        description="Port to bind the server to. Default is 'auto', which tries to "
        "bind to port 19000 and increments if the port is already in use.",
    )

    _search: Search = PrivateAttr()
    _app: web.Application = PrivateAttr()
    _site: web.TCPSite | None = PrivateAttr(default=None)
    _open_websockets: WeakSet[web.WebSocketResponse] = PrivateAttr(
        default_factory=WeakSet
    )

    async def prepare(self, search: Search) -> None:
        self._search = search
        self._app = web.Application(logger=logger)
        self.add_routes()

    def add_routes(self) -> None:
        router = self._app.router
        router.add_get("/", self.get_index)
        router.add_get("/api/v1/detections", self.get_detections)
        router.add_get("/api/v1/receivers", self.get_receivers)
        router.add_get("/api/v1/search", self.get_search)

        # Configure default CORS settings.
        cors = aiohttp_cors.setup(
            self._app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        for route in list(router.routes()):
            try:
                cors.add(route)
            except ValueError as exc:
                logger.exception(exc)
                continue

    async def get_index(self, request: web.Request) -> web.Response:
        qseek_version = version("qseek")
        return web.Response(text=f"{qseek_version} HTTP server running...")

    async def get_detections(
        self, request: web.Request, n_detections: int = -1
    ) -> web.Response:
        async with aiofiles.open(self._search._catalog.detections_file, "r") as f:
            lines = await f.readlines()
        data = "{" + lines + "}"
        await web.Response(text=data, content_type="application/json")

    async def get_receivers(
        self, request: web.Request, n_detections: int = -1
    ) -> web.Response:
        async with aiofiles.open(self._search._catalog.receivers_file, "r") as f:
            lines = await f.readlines()
        data = "{" + lines + "}"
        await web.Response(text=data, content_type="application/json")

    async def get_search(self, request: web.Request) -> web.Response:
        return web.Response(
            self._search.model_dump_json(),
            content_type="application/json",
        )

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._open_websockets.add(ws)
        logger.info("New WebSocket connection established: %s", request.remote)

        for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                # Handle incoming messages from the client if needed
                pass
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(
                    "WebSocket connection closed with exception %s", ws.exception()
                )

    async def new_detections(self, detections: list[EventDetection]) -> None:
        if not self._open_websockets:
            return

        data = {
            "type": "new_detections",
            "detections": [d.model_dump(exclude={"receivers"}) for d in detections],
            "receivers": [d.receivers.model_dump() for d in detections],
        }

        for ws in self._open_websockets:
            await ws.send_json(data)

    async def start(self) -> None:
        runner = web.AppRunner(self._app)
        await runner.setup()

        port = 19000 if self.port == "auto" else self.port

        for _ in range(MAX_TRIES):
            try:
                site = web.TCPSite(runner, str(self.host), port)
                await site.start()
                logger.info("started on http://%s:%d", self.host, port)
                self._site = site
                break
            except OSError:
                if self.port != "auto":
                    raise
                logger.warning("port %d is in use, trying next port...", port)
                port += 1
        else:
            raise OSError(f"Could not bind to a port after {MAX_TRIES} tries")

    async def stop(self) -> None:
        logger.info("shutting down HTTP server...")
        if self._site:
            await self._site.stop()
        await self._app.shutdown()
        await self._app.cleanup()
