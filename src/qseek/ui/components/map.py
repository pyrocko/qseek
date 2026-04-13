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
            with m:
                ui.run_javascript(
                    f"""
                const map = getElement({m.id}).map;
                const data = {json.dumps(marker_data)};
                const canvasRenderer = L.canvas(); // Use canvas for high-performance rendering

                var group = L.featureGroup();
                data.forEach(point => {{
                    L.circleMarker([point[0], point[1]], {{
                        renderer: canvasRenderer,
                        radius: point[2] * 4,
                        stroke: false,
                        fillColor: point[3],
                        fillOpacity: 0.7
                    }}).on('click', () => window.location.href = '/event/' + point[4])
                    .addTo(group);
                }});
                group.addTo(map);
                map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
                """,
                )

        background_tasks.create(add_markers(), name="markers-map")


class NetworkMap(Component):
    name = "Network Map"
    description = """Map of seismic stations in the network."""

    async def view(self) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()

        rows = []
        for ev in catalog.events:
            for receiver in ev.event.receivers:
                if not receiver.phase_arrivals:
                    continue
                rows.append(
                    {
                        "station": receiver.nsl.pretty_str(strip=True),
                        "station_lat": receiver.lat,
                        "station_lon": receiver.lon,
                    }
                )
        unique_stations = {row["station"]: row for row in rows}.values()
        station_data = [
            {"label": s["station"], "lat": s["station_lat"], "lon": s["station_lon"]}
            for s in unique_stations
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

        # Match station styling used in event station map.
        station_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
            '<polygon points="8,0.5 14.5,11.75 1.5,11.75" '
            'fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
            "</svg>"
        )

        ui.run_javascript(
            f"""
            const map = getElement({m.id}).map;
            const stations = {json.dumps(station_data)};

            const stationIcon = L.divIcon({{
                html: '{station_svg}',
                iconSize: [16, 16],
                iconAnchor: [8, 8],
                className: ''
            }});

            const group = L.featureGroup();
            stations.forEach((s) => {{
                group.addLayer(
                    L.marker([s.lat, s.lon], {{ icon: stationIcon }})
                        .on('click', () => {{
                            window.location.href = '/station/' + encodeURIComponent(s.label);
                        }})
                        .bindTooltip(`<b>${{s.label}}</b>`, {{ permanent: false }})
                );
            }});
            group.addTo(map);

            if (stations.length > 0) {{
                map.fitBounds(group.getBounds(), {{ padding: [30, 30] }});
            }}
            """
        )
