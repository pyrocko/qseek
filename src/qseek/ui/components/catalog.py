from __future__ import annotations

from datetime import datetime
from uuid import UUID

import plotly.express as px
from nicegui import ui
from nicegui.events import GenericEventArguments
from pydantic import BaseModel

from qseek.models.catalog import EventCatalog
from qseek.models.detection import EventDetection
from qseek.search import Search
from qseek.ui.base import ROUTER, FrontendPage
from qseek.ui.elements import panel


class SimpleEvent(BaseModel):
    uid: UUID
    time: datetime
    lat: float
    lon: float
    depth: float
    semblance: float

    @classmethod
    def from_event(cls, event: EventDetection):
        return cls(
            uid=event.uid,
            time=event.time,
            lat=event.lat,
            lon=event.lon,
            depth=event.depth,
            semblance=event.semblance,
        )


class SimpleCatalog(BaseModel):
    catalog: EventCatalog

    def to_rows(self, start: int, count: int = 30):
        if count > 0:
            events = self.catalog.events[start : start + count]
        else:
            events = self.catalog.events
        return [SimpleEvent.from_event(event).model_dump() for event in events]


class CatalogTable:
    def __init__(self, search: Search) -> None:
        self.search = search

    def ui(self) -> None:
        columns = [
            {
                "field": "time",
                "name": "time",
                "label": "Time",
                "align": "left",
                "type": "datetime",
            },
            {
                "field": "lat",
                "name": "lat",
                "label": "Latitude",
                "type": "float",
            },
            {
                "field": "lon",
                "name": "lon",
                "label": "Longitude",
                "type": "float",
            },
            {
                "field": "depth",
                "name": "depth",
                "label": "Depth",
                "type": "float",
                ":format": "value => (value / 1e3).toFixed(1) + ' km'",
            },
            {
                "field": "semblance",
                "name": "semblance",
                "label": "Semblance",
                "type": "float",
                ":format": "value => value.toFixed(3)",
            },
        ]

        cat = SimpleCatalog(catalog=self.search.catalog).to_rows(start=0, count=0)

        table = ui.table(columns=columns, rows=cat, row_key="uid", pagination=25)
        table.classes("w-full shadow-none")

        def show_event_details(event: GenericEventArguments):
            uid = event.args[1]["uid"]
            ROUTER.open(f"/event/{uid}")

        table.on("row-click", show_event_details)


class CatalogMap:
    def __init__(self, search: Search) -> None:
        self.search = search
        self.catalog = search.catalog

    def map(self):
        catalog = self.catalog
        center = catalog.events[0].lat, catalog.events[0].lon
        map = ui.leaflet(center=center, zoom=10)
        map.classes("h-[20vw]")
        map.tile_layer(
            url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x})",
            options={
                "attribution": "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community"
            },
        )
        for ev in catalog:
            map.generic_layer(
                name="circle",
                args=[
                    (ev.effective_lat, ev.effective_lon),
                    {
                        "fillColor": "red",
                        "fillOpacity": 0.4,
                        "stroke": False,
                        "radius": ev.semblance * 40,
                    },
                ],
            )


class CatalogRate:
    def __init__(self, search: Search) -> None:
        self.search = search
        self.catalog = search.catalog

    def ui(self):
        catalog = self.catalog

        fig = px.scatter(
            x=[ev.time for ev in catalog.events],
            y=[ev.semblance for ev in catalog.events],
            size=[ev.semblance for ev in catalog.events],
            labels={"x": "Time", "y": "Semblance"},
            color_discrete_sequence=["black"],
            render_mode="webgl",
        )
        fig.update_traces(marker={"line": {"width": 0}, "opacity": 0.4})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        ui.plotly(fig).classes("w-full h-full")


class CatalogView(FrontendPage):
    path = "/events"
    name = "Events"
    icon = "event"

    def ui(self) -> None:
        with ui.row(wrap=True):
            with ui.column(wrap=True).classes("col-grow h-auto"):
                with panel("w-full", title="Seismicity Rate"):
                    CatalogRate(self.search).ui()
                with panel("w-full", title="Event Map"):
                    CatalogMap(self.search).map()
            with panel("col col-md-4", title="Event List"):
                CatalogTable(self.search).ui()
