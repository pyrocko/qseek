import json

from nicegui import background_tasks, ui

from qseek.models.station import Station
from qseek.ui.base import Component

_STATION_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
    '<polygon points="10,1 19,17 1,17"'
    ' fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
    "</svg>"
)


class StationComponent(Component):
    name: str = "Station Component"
    description: str = ""

    def __init__(self, station: Station) -> None:
        self.station = station

    async def view(self) -> None:
        raise NotImplementedError


class StationMap(StationComponent):
    name = "Location"
    description = "Map showing the station's geographic position."

    async def view(self) -> None:
        station = self.station
        m = ui.leaflet(
            center=(station.effective_lat, station.effective_lon), zoom=10
        ).classes("w-full h-80 rounded-lg shadow")
        m.clear_layers()
        m.tile_layer(
            url_template="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            options={
                "attribution": (
                    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    " contributors"
                    ' &copy; <a href="https://carto.com/attributions">CARTO</a>'
                ),
                "subdomains": "abcd",
                "maxZoom": 20,
            },
        )
        await m.initialized()

        async def add_marker():
            tip = f"<b>{station.nsl.pretty}</b><br>Elevation: {station.elevation:.0f} m"
            if station.depth > 0:
                tip += f"<br>Depth: {station.depth:.0f} m"
            marker_info = json.dumps(
                {
                    "lat": station.effective_lat,
                    "lon": station.effective_lon,
                    "tip": tip,
                }
            )
            with m:
                ui.run_javascript(
                    f"""
                const map = getElement({m.id}).map;
                const s = {marker_info};
                const icon = L.divIcon({{
                    html: '{_STATION_SVG}',
                    iconSize: [20, 20],
                    iconAnchor: [10, 17],
                    className: ''
                }});
                L.marker([s.lat, s.lon], {{icon: icon}})
                    .bindTooltip(s.tip, {{permanent: true, direction: 'top', offset: [0, -18]}})
                    .addTo(map);
                """,
                )

        background_tasks.create(add_marker(), name="station-page-marker")


class StationDetails(StationComponent):
    name = "Station details"
    description = ""

    async def view(self) -> None:
        station = self.station

        rows = [
            {"property": "NSL", "value": station.nsl.pretty},
            {"property": "Network", "value": station.network},
            {"property": "Station", "value": station.station},
            {"property": "Location", "value": station.location or "—"},
            {"property": "Latitude", "value": f"{station.effective_lat:.6f}°"},
            {"property": "Longitude", "value": f"{station.effective_lon:.6f}°"},
            {"property": "Elevation", "value": f"{station.elevation:,.1f} m"},
        ]
        if station.depth > 0:
            rows += [
                {"property": "Depth", "value": f"{station.depth:,.1f} m"},
                {
                    "property": "Effective elevation",
                    "value": f"{station.effective_elevation:,.1f} m",
                },
            ]
        if station.north_shift or station.east_shift:
            rows += [
                {"property": "North shift", "value": f"{station.north_shift:,.1f} m"},
                {"property": "East shift", "value": f"{station.east_shift:,.1f} m"},
            ]

        columns = [
            {
                "name": "property",
                "label": "Property",
                "field": "property",
                "align": "left",
            },
            {"name": "value", "label": "Value", "field": "value", "align": "left"},
        ]
        table = (
            ui.table(columns=columns, rows=rows, row_key="property")
            .classes("w-full text-sm")
            .props("dense flat bordered hide-header")
        )
        table.add_slot(
            "body-row",
            """
            <q-tr :props="props" :class="props.rowIndex % 2 === 0 ? 'bg-grey-1' : ''">
                <q-td class="text-grey-6 text-weight-medium" style="width:200px">
                    {{ props.row.property }}
                </q-td>
                <q-td class="font-mono">{{ props.row.value }}</q-td>
            </q-tr>
            """,
        )
