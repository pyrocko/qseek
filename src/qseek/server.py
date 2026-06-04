from __future__ import annotations

import asyncio
import logging
from importlib.metadata import version
from ipaddress import IPv4Address
from typing import TYPE_CHECKING, Literal
from weakref import WeakSet

import aiohttp_cors
from aiohttp import web
from pydantic import BaseModel, Field, PrivateAttr

from qseek.base import Model
from qseek.models.detection import EventDetection

if TYPE_CHECKING:
    from qseek.search import Search

logger = logging.getLogger(__name__)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

MAX_TRIES = 5


class WebsocketMessage(BaseModel):
    type: Literal["NewDetections"] = Field(..., description="Type of the message")
    data: DetectionCollection


class DetectionCollection(BaseModel):
    detections: list[EventDetection]

    @property
    def n_detections(self) -> int:
        return len(self.detections)


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
        router.add_get("/api/v1/search", self.get_search)
        router.add_get("/api/v1/catalog", self.get_catalog)
        router.add_get("/api/v1/detections", self.get_detections)
        router.add_get("/api/v1/detections_receivers", self.get_receivers)
        router.add_get("/api/v1/ws", self.websocket_handler)

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
            cors.add(route)

    async def get_index(self, request: web.Request) -> web.Response:
        qseek_version = version("qseek")
        return web.Response(text=f"{qseek_version} HTTP server running...")

    async def get_search(self, request: web.Request) -> web.Response:
        return web.Response(
            body=self._search.model_dump_json(),
            content_type="application/json",
        )

    async def get_catalog(self, request: web.Request) -> web.Response:
        catalog = self._search._catalog
        return web.Response(
            body=catalog.model_dump_json(exclude={"events"}),
            content_type="application/json",
        )

    async def get_detections(self, request: web.Request) -> web.Response:
        catalog = self._search._catalog
        return web.FileResponse(
            catalog.detections_file,
            headers={"Content-Type": "text/plain"},
        )

    async def get_receivers(self, request: web.Request) -> web.Response:
        catalog = self._search._catalog
        return web.FileResponse(
            catalog.receivers_file,
            headers={"Content-Type": "text/plain"},
        )

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._open_websockets.add(ws)
        logger.info("new WebSocket connection established to %s", request.remote)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle incoming messages from the client if needed
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(
                        "WebSocket connection closed with exception %s", ws.exception()
                    )
                elif msg.data == "close":
                    logger.info(
                        "WebSocket connection closed by client: %s", request.remote
                    )
                    await ws.close()
                    break
        finally:
            self._open_websockets.discard(ws)
            logger.info("WebSocket connection closed: %s", request.remote)

        return ws

    async def new_detections(self, detections: list[EventDetection]) -> None:
        logger.info("New detections received, sending to WebSocket clients...")
        if not self._open_websockets:
            logger.info(
                "No WebSocket clients connected, skipping sending new detections"
            )
            return

        websocket_message = WebsocketMessage(
            type="NewDetections",
            data=DetectionCollection(detections=detections),
        )

        data = await asyncio.to_thread(websocket_message.model_dump_json)
        for ws in self._open_websockets:
            try:
                await ws.send_str(data)
            except Exception:
                logger.exception(
                    "Error sending WebSocket message to client, closing connection"
                )

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
