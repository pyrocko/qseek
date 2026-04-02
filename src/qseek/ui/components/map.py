import numpy as np
from nicegui import ui

from qseek.ui.base import Component

import matplotlib.cm as cm
import matplotlib.colors as mcolors


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

        def depth_to_color(d):
            return mcolors.to_hex(cmap(norm(d)))

        for lat, lon, depth, magnitude in zip(
            lats, lons, depths, magnitudes, strict=True
        ):
            color = depth_to_color(depth)
            size = magnitude * 4  # Adjust size based on magnitude

            m.generic_layer(
                name="circleMarker",
                args=[
                    (lat, lon),
                    {
                        "radius": size,
                        "color": color,
                        "fillColor": color,
                        "fillOpacity": 0.8,
                    },
                ],
            )
