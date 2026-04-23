import asyncio
import json

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from nicegui import background_tasks, ui

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state


class OverviewMap(Component):
    name = "Event Map"
    description = """Map of detected events. Color corresponds to depth and size corresponds to magnitude."""

    async def view(self) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()
        sorted_events = sorted(catalog.events, key=lambda ev: ev.depth)

        norm = mcolors.Normalize(vmin=min(catalog.depths), vmax=max(catalog.depths))
        cmap = cm.get_cmap("magma_r")

        norm_depths = norm(np.array([ev.depth for ev in sorted_events]))
        colors = [mcolors.to_hex(cmap(d)) for d in norm_depths]

        marker_data = [
            [float(ev.lat), float(ev.lon), float(ev.semblance), color, str(ev.uid)]
            for ev, color in zip(sorted_events, colors, strict=True)
        ]

        m = ui.leaflet(center=(np.mean(catalog.lats), np.mean(catalog.lons))).classes(
            "w-full h-96 rounded-lg shadow"
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

        await m.initialized()

        async def add_markers():
            data = await asyncio.to_thread(json.dumps, marker_data)
            with m:
                ui.run_javascript(
                    f"""
                const map = getElement({m.id}).map;
                const data = {data};
                const canvasRenderer = L.canvas(); // Use canvas for high-performance rendering

                var group = L.featureGroup();
                data.forEach(point => {{
                    L.circleMarker([point[0], point[1]], {{
                        renderer: canvasRenderer,
                        radius: point[2] * 4,
                        stroke: false,
                        fillColor: point[3],
                        fillOpacity: 0.7
                    }}).on('click', () => window.location.href = 'event/' + point[4])
                    .addTo(group);
                }});
                group.addTo(map);
                map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
                """,
                )

        async def add_stations():
            search = await state.run.get_search()
            station_data = [
                {
                    "lat": float(sta.effective_lat),
                    "lon": float(sta.effective_lon),
                    "label": sta.nsl.pretty,
                    "elevation": sta.elevation,
                    "depth": sta.depth,
                }
                for sta in search.stations
            ]
            data = await asyncio.to_thread(json.dumps, station_data)
            station_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" opacity="0.75">'
                '<polygon points="8,0.5 14.5,11.75 1.5,11.75"'
                ' fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
                "</svg>"
            )
            with m:
                ui.run_javascript(
                    f"""
                const map = getElement({m.id}).map;
                const stations = {data};
                const stationIcon = L.divIcon({{
                    html: '{station_svg}',
                    iconSize: [12, 12],
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
                        .addTo(map);
                }});
                """,
                )

        background_tasks.create(add_markers(), name="markers-map")
        background_tasks.create(add_stations(), name="stations-map")
