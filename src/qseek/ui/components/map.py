import asyncio
import json

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from nicegui import background_tasks, ui

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import EVENT_ANIMATED_SVG


class OverviewMap(Component):
    name = "Event Map"
    description = """Map of detected events. Color corresponds to depth and size corresponds to magnitude."""

    async def view(
        self,
        show_events: bool = True,
        marker_colors: list[str] | None = None,
    ) -> None:
        state = get_tab_state()
        catalog = self.catalog
        cmap = cm.get_cmap("magma_r")

        m = ui.leaflet(center=(np.mean(catalog.lats), np.mean(catalog.lons))).classes(
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

        await m.initialized()
        with m:
            ui.run_javascript(
                f"""
                const map = getElement({m.id}).map;
                L.control.scale().addTo(map);
                map._canvasRenderer = L.canvas();
                map._eventGroup = L.featureGroup().addTo(map);
                map._stationGroup = L.featureGroup().addTo(map);
                """,
            )

        async def update_markers():
            if not show_events:
                return

            events = catalog.events
            if not events:
                return

            depths = np.array([ev.depth for ev in events])
            norm = mcolors.Normalize(vmin=depths.min(), vmax=depths.max())
            colors = (
                [mcolors.to_hex(cmap(norm(d))) for d in depths]
                if marker_colors is None
                else marker_colors
            )
            marker_data = [
                [float(ev.lat), float(ev.lon), float(ev.semblance), color, str(ev.uid)]
                for ev, color in zip(events, colors, strict=True)
            ]
            marker_data = sorted(
                marker_data, key=lambda x: x[2]
            )  # sort by depth for better layering
            latest = max(catalog.events, key=lambda ev: ev.time)
            latest_data = [float(latest.lat), float(latest.lon), str(latest.uid)]
            data, latest_json = await asyncio.to_thread(
                lambda: (json.dumps(marker_data), json.dumps(latest_data))
            )
            with m:
                ui.run_javascript(
                    f"""
                    const map = getElement({m.id}).map;
                    map._eventGroup.clearLayers();
                    const data = {data};
                    data.forEach(point => {{
                        L.circleMarker([point[0], point[1]], {{
                            renderer: map._canvasRenderer,
                            radius: point[2] * 4,
                            stroke: false,
                            fillColor: point[3],
                            fillOpacity: 0.7
                        }}).on('click', () => window.location.href = 'event/' + point[4])
                        .addTo(map._eventGroup);
                    }});
                    """
                )
                if state.run.live:
                    ui.run_javascript(
                        f"""
                        const map = getElement({m.id}).map;
                        const latest = {latest_json};
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

        async def add_stations():
            search = await state.run.get_search()
            station_data = [
                {
                    "lat": float(sta.effective_lat),
                    "lon": float(sta.effective_lon),
                    "label": sta.nsl.pretty_str(strip=True),
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
                        .addTo(map._stationGroup);
                }});
                """,
                )

        async def update_map():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(update_markers())
                tg.create_task(add_stations())
            with m:
                ui.run_javascript(
                    f"""
                    const map = getElement({m.id}).map;
                    const b = map._eventGroup.getBounds();
                    if (b.isValid()) {{
                        map.fitBounds(b, {{padding: [20, 20]}});
                    }}
                    else {{
                        const b = map._stationGroup.getBounds();
                        if (b.isValid()) {{
                            map.fitBounds(b, {{padding: [20, 20]}});
                        }}
                    }}
                    """,
                )

        self.catalog.updated.subscribe(update_markers)
        background_tasks.create(update_map(), name="update-map")
