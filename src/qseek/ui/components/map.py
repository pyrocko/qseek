import json

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from nicegui import ui

from qseek.ui.base import Component


class OverviewMap(Component):
    name = "Event Map"
    description = """Map of detected events. Color corresponds to depth and size corresponds to magnitude."""

    async def view(self) -> None:
        catalog = await self.run.get_catalog()
        lats = catalog.lats
        lons = catalog.lons
        depths = catalog.depths
        magnitudes = catalog.magnitudes

        m = ui.leaflet(center=(np.mean(lats), np.mean(lons)), zoom=9).classes(
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

        norm = mcolors.Normalize(vmin=min(depths), vmax=max(depths))
        cmap = cm.get_cmap("magma")
        norm_depths = norm(np.array(depths))
        colors = [mcolors.to_hex(cmap(d)) for d in norm_depths]

        marker_data = [
            [float(lat), float(lon), float(semblance * 4), color]
            for lat, lon, semblance, color in zip(
                lats, lons, catalog.semblances, colors, strict=True
            )
        ]

        await m.initialized()

        ui.run_javascript(
            f"""
            const map = getElement({m.id}).map;
            const data = {json.dumps(marker_data)};
            const canvasRenderer = L.canvas(); // Use canvas for high-performance rendering

            data.forEach(point => {{
                L.circleMarker([point[0], point[1]], {{
                    renderer: canvasRenderer,
                    radius: point[2],
                    color: point[3],
                    fillColor: point[3],
                    fillOpacity: 0.8
                }}).addTo(map);
            }});
            """
        )
