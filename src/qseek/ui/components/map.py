import numpy as np
from nicegui import ui

from qseek.ui.base import Component

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json


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

        # Farben vektorisiert berechnen
        norm = mcolors.Normalize(vmin=min(depths), vmax=max(depths))
        cmap = cm.get_cmap("magma")
        norm_depths = norm(np.array(depths))
        colors = [mcolors.to_hex(cmap(d)) for d in norm_depths]

        # Marker-Daten vorbereiten
        markers_data = [
            {
                "lat": float(lat),
                "lon": float(lon),
                "radius": float(magnitude * 4),
                "color": color,
            }
            for lat, lon, magnitude, color in zip(
                lats, lons, magnitudes, colors, strict=True
            )
        ]

        # Warten bis die Map im Browser initialisiert ist
        await m.initialized()

        # Alle Marker gebündelt in einem JS-Aufruf
        ui.run_javascript(f"""
            var map = getElement({m.id}).map;
            var data = {json.dumps(markers_data)};
            var group = L.layerGroup();
            for (var i = 0; i < data.length; i++) {{
                var d = data[i];
                L.circleMarker([d.lat, d.lon], {{
                    radius: d.radius,
                    color: d.color,
                    fillColor: d.color,
                    fillOpacity: 0.8
                }}).addTo(group);
            }}
            group.addTo(map);
        """)
