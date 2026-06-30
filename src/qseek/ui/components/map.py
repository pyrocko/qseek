from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Iterable

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.pyplot import get_cmap
from nicegui import background_tasks, ui

from qseek.ui.base import Panel
from qseek.ui.models import EventMinimal
from qseek.ui.state import CatalogStore
from qseek.ui.utils import EVENT_ANIMATED_SVG

if TYPE_CHECKING:
    from nicegui.elements.leaflet import Leaflet

    from qseek.models.station import Station


class OverviewMap(Panel):
    title = "Event Map"
    description = """
Map of detected events. Color corresponds to depth and size corresponds to magnitude.
"""
    _map: Leaflet | None = None
    _initialized = False
    _events: list[EventMinimal]

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
    ) -> None:
        super().__init__()
        self._events = []
        self._cmap_norm = mcolors.Normalize(vmin=0, vmax=0)

        with self:
            m = ui.leaflet(center=(center_lat, center_lon)).classes(
                "w-full h-128 rounded-lg shadow"
            )
            m.clear_layers()
            m.tile_layer(
                url_template="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                options={
                    "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                    "subdomains": "abcd",
                    "maxZoom": 20,
                },
            )
            self._map = m

    async def initialize(self):
        if self._initialized:
            return
        await self._map.initialized()
        with self._map as m:
            ui.run_javascript(
                f"""
                const map = getElement({m.id}).map;
                L.control.scale().addTo(map);
                map._canvasRenderer = L.canvas();
                map._eventData = [];
                map._eventGroup = L.featureGroup().addTo(map);
                map._stationGroup = L.featureGroup().addTo(map);
                """,
            )
            self._initialized = True

    async def add_stations(self, stations: Iterable[Station]):
        station_data = [
            {
                "lat": float(sta.effective_lat),
                "lon": float(sta.effective_lon),
                "label": sta.nsl.pretty_str(strip=True),
                "elevation": sta.elevation,
                "depth": sta.depth,
            }
            for sta in stations
        ]
        data = await asyncio.to_thread(json.dumps, station_data)
        station_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" opacity="0.75">'
            '<polygon points="8,0.5 14.5,11.75 1.5,11.75"'
            ' fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
            "</svg>"
        )
        with self._map as m:
            ui.run_javascript(
                f"""
                const map = getElement({m.id}).map;
                const stations = {data};
                const stationIcon = L.divIcon({{
                    html: '{station_svg}',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8],
                    className: ''
                }});
                stations.forEach(function(s) {{
                    var tip = '<b>' + s.label + '</b>'
                        + '<br>Elevation: ' + s.elevation.toFixed(0) + ' m';
                    if (s.depth > 0) tip += '<br>Depth: ' + s.depth.toFixed(0) + ' m';
                    L.marker([s.lat, s.lon], {{icon: stationIcon}})
                        .bindTooltip(tip, {{permanent: false}})
                        .on('click', function() {{ window.location.href = '/station/' + s.label; }})
                        .addTo(map._stationGroup);
                }});
                """,
            )

    async def add_event_markers(
        self,
        events: list[EventMinimal],
        cmap: str = "magma_r",
        marker_colors: list[str] | None = None,
        highlight_latest: bool = True,
        clear_markers: bool = True,
        update_extent: bool = True,
    ):
        if marker_colors is None:
            mpl_cmap = get_cmap(cmap)
            depths = np.array([ev.depth for ev in events])
            norm = self._cmap_norm
            norm.vmin = min(norm.vmin, depths.min())
            norm.vmax = max(norm.vmax, depths.max())
            marker_colors = (
                [mcolors.to_hex(mpl_cmap(norm(d))) for d in depths]
                if marker_colors is None
                else marker_colors
            )

        marker_data = [
            [ev.lat, ev.lon, ev.depth, ev.semblance, color, str(ev.uid)]
            for ev, color in zip(events, marker_colors, strict=True)
        ]

        data = await asyncio.to_thread(json.dumps, marker_data)
        with self._map as m:
            ui.run_javascript(
                f"""
                const map = getElement({m.id}).map;
                if ({int(clear_markers)}) {{
                    map._eventGroup.clearLayers();
                    map._eventData = [];
                }}
                map._eventData.push(...{data});
                map._eventData.sort((a, b) => a[2] - b[2]); // sort by depth

                const _maxSemblance = Math.max(...map._eventData.map(p => p[3]), 1e-9);
                map._eventData.forEach(point => {{
                    L.circleMarker([point[0], point[1]], {{
                        renderer: map._canvasRenderer,
                        radius: (point[3] / _maxSemblance) * 4,
                        stroke: false,
                        fillColor: point[4],
                        fillOpacity: 0.7
                    }}).on('click', () => window.location.href = 'event/' + point[5])
                    .addTo(map._eventGroup);
                }});
                """
            )
            if highlight_latest:
                latest = max(events, key=lambda ev: ev.time)
                latest_data = [float(latest.lat), float(latest.lon), str(latest.uid)]
                ui.run_javascript(
                    f"""
                    const map = getElement({m.id}).map;
                    const latest = {json.dumps(latest_data)};
                    const latestIcon = L.divIcon({{
                        html: '{EVENT_ANIMATED_SVG}',
                        iconSize: [40, 40],
                        iconAnchor: [20, 20],
                        className: ''
                    }});
                    L.marker([latest[0], latest[1]], {{icon: latestIcon}})
                        .on('click', () => window.location.href = 'event/' + latest[2])
                        .addTo(map._eventGroup);
                    """
                )

        if update_extent:
            await self.update_extent()

    async def update_extent(self):
        with self._map as m:
            ui.run_javascript(
                f"""
                    const map = getElement({m.id}).map;
                    const b = map._eventGroup.getBounds();
                    if (b.isValid()) {{
                        map.fitBounds(b, {{padding: [20, 20]}});
                    }} else {{
                        const b = map._stationGroup.getBounds();
                        if (b.isValid()) {{
                            map.fitBounds(b, {{padding: [20, 20]}});
                        }}
                    }}
                    """,
            )

    async def add_catalog(
        self,
        catalog: CatalogStore,
        stations: list[Station] | None = None,
        show_latest: bool = True,
    ):
        await self.initialize()
        background_tasks.create(
            self.add_event_markers(catalog.events, highlight_latest=show_latest)
        )
        if stations is not None:
            background_tasks.create(self.add_stations(stations))

    def attach_catalog(self, catalog: CatalogStore):
        async def update(highlight_latest: bool = True):
            await self.add_event_markers(
                catalog.events,
                highlight_latest=highlight_latest,
                clear_markers=True,
                update_extent=False,
            )

        async def update_no_highlight():
            await update(highlight_latest=False)

        catalog.new_events.subscribe(update)
        catalog.updated.subscribe(update_no_highlight)
