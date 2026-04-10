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
                    }}).on('click', () => window.location.href = 'event/' + point[4])
                    .addTo(group);
                }});
                group.addTo(map);
                map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
                """,
                )

        background_tasks.create(add_markers(), name="markers-map")
