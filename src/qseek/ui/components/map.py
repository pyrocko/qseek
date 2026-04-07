import json

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from nicegui import ui

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state


class OverviewMap(Component):
    name = "Event Map"
    description = """Map of detected events. Color corresponds to depth and size corresponds to magnitude."""

    async def view(self) -> None:
        state = get_tab_state()
        catalog = await state.run.get_catalog()

        norm = mcolors.Normalize(vmin=min(catalog.depths), vmax=max(catalog.depths))
        cmap = cm.get_cmap("magma")
        norm_depths = norm(np.array(catalog.depths))
        colors = [mcolors.to_hex(cmap(d)) for d in norm_depths]

        marker_data = [
            [float(lat), float(lon), float(semblance), color, str(uid)]
            for lat, lon, semblance, color, uid in zip(
                catalog.lats,
                catalog.lons,
                catalog.semblances,
                colors,
                catalog.uids,
                strict=True,
            )
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

        ui.run_javascript(
            f"""
            const map = getElement({m.id}).map;
            const data = {json.dumps(marker_data)};
            const canvasRenderer = L.canvas(); // Use canvas for high-performance rendering

            var group = L.featureGroup();
            data.forEach(point => {{
                L.circleMarker([point[0], point[1]], {{
                    renderer: canvasRenderer,
                    radius: point[2] * 6,
                    stroke: false,
                    fillColor: point[3],
                    fillOpacity: 0.7
                }}).on('click', () => window.location.href = 'event/' + point[4])
                  .addTo(group);
            }});
            group.addTo(map);
            map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
            """
        )
