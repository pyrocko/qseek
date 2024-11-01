from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AbstractSet, Any, Mapping, Type, TypeVar
from uuid import UUID

import aiohttp_cors
from aiohttp import web
from pkg_resources import get_distribution
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from qseek.search import Search

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
MAX_TRIES = 5

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)
AbstractSetIntStr = AbstractSet[int | str]
MappingIntStrAny = Mapping[int | str, Any]


class DetectionRequest(BaseModel):
    uid: UUID


async def parse_pydantic_request(
    request: web.Request, model: Type[TBaseModel]
) -> TBaseModel:
    """Parse a aiohttp request and return a pydantic model.

    Raises:
        web.HTTPMethodNotAllowed: If method is not GET/POST/PUT
        web.HTTPBadRequest: _description_
        web.HTTPError: _description_

    Returns:
        TBaseModel: Parsed pydantic model.
    """
    try:
        if request.method == "GET":
            query = {}
            for key, value in request.query.items():
                if key.endswith("[]"):
                    key = key.strip("[]")
                    if key not in query:
                        query[key] = []
                    query[key].append(value)
                else:
                    query[key] = value

            data = model.model_validate(query)
        elif request.method in ("POST", "PUT"):
            data = model.model_validate(await request.post())
        else:
            raise web.HTTPMethodNotAllowed(
                request.method, allowed_methods=("GET", "POST", "PUT")
            )
    except ValidationError as exc:
        raise web.HTTPBadRequest(
            text=exc.json(), content_type="application/json"
        ) from exc
    except Exception as exc:
        raise web.HTTPError(text="Cannot parse request") from exc
    return data


def pydantic_response(
    model: BaseModel,
    headers: dict[str, Any] | None = None,
    exclude: dict | None = None,
) -> web.Response:
    try:
        return web.json_response(
            text=model.model_dump_json(exclude=exclude).replace("NaN", "null"),
            headers=headers,
        )
    except TypeError as exc:
        return web.HTTPServerError(reason=str(exc))


class WebServer:
    def __init__(self, search: Search) -> None:
        self.search = search
        self.app = web.Application(logger=logger)
        self.add_routes()

    def add_routes(self) -> None:
        router = self.app.router
        router.add_get("/", self.get_index)
        router.add_get("/api/v1/detections", self.get_detections)
        router.add_get("/api/v1/search", self.get_search)
        router.add_get(r"/api/v1/detection/{uid}", self.get_detection)

        # Configure default CORS settings.
        cors = aiohttp_cors.setup(
            self.app,
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
        version = get_distribution("qseek")
        return web.Response(text=f"{version} HTTP server running...")

    async def get_detections(self, request: web.Request) -> web.Response:
        return pydantic_response(
            self.search._catalog,
            exclude={"detections": {"__all__": {"octree", "stations"}}},
        )

    async def get_detection(self, request: web.Request) -> web.Response:
        req = await parse_pydantic_request(request, model=DetectionRequest)
        try:
            detection = self.search._catalog.get(req.uid)
        except KeyError:
            return web.HTTPNotFound()
        return pydantic_response(detection)

    async def get_search(self, request: web.Request) -> web.Response:
        return pydantic_response(self.search)

    async def start(self) -> None:
        runner = web.AppRunner(self.app)
        await runner.setup()

        port = 3445

        for _ in range(MAX_TRIES):
            try:
                site = web.TCPSite(runner, "0.0.0.0", port)
                await site.start()
                logger.info("started on http://0.0.0.0:%d", port)
                break
            except OSError:
                port += 1
