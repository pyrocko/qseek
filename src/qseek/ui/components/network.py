import asyncio
import json

from nicegui import background_tasks, ui

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state

_STATION_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
    '<polygon points="8,0.5 14.5,11.75 1.5,11.75"'
    ' fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
    "</svg>"
)


class StationMap(Component):
    name = "Station Map"
    description = "Map of all stations in the network inventory."

    async def view(self) -> None:
        state = get_tab_state()
        search = await state.run.get_search()
        stations = list(search.stations)

        if not stations:
            ui.label("No stations in inventory.").classes("text-grey-6 italic")
            return

        center_lat = sum(s.effective_lat for s in stations) / len(stations)
        center_lon = sum(s.effective_lon for s in stations) / len(stations)

        m = ui.leaflet(center=(center_lat, center_lon)).classes(
            "w-full h-96 rounded-lg shadow"
        )
        m.clear_layers()
        m.tile_layer(
            url_template="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            options={
                "attribution": (
                    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    ' contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                ),
                "subdomains": "abcd",
                "maxZoom": 20,
            },
        )

        await m.initialized()

        async def add_markers():
            station_data = [
                {
                    "lat": float(sta.effective_lat),
                    "lon": float(sta.effective_lon),
                    "label": sta.nsl.pretty,
                    "elevation": sta.elevation,
                    "depth": sta.depth,
                }
                for sta in stations
            ]
            data = await asyncio.to_thread(json.dumps, station_data)
            with m:
                ui.run_javascript(
                    f"""
                const map = getElement({m.id}).map;
                const stations = {data};
                const stationIcon = L.divIcon({{
                    html: '{_STATION_SVG}',
                    iconSize: [16, 16],
                    iconAnchor: [8, 12],
                    className: ''
                }});
                var group = L.featureGroup();
                stations.forEach(function(s) {{
                    var tip = '<b>' + s.label + '</b>'
                        + '<br>Elevation: ' + s.elevation.toFixed(0) + ' m';
                    if (s.depth > 0) tip += '<br>Depth: ' + s.depth.toFixed(0) + ' m';
                    group.addLayer(
                        L.marker([s.lat, s.lon], {{icon: stationIcon}})
                            .bindTooltip(tip, {{permanent: false}})
                            .on('click', function() {{ window.location.href = '/station/' + s.label; }})
                    );
                }});
                group.addTo(map);
                map.fitBounds(group.getBounds(), {{padding: [30, 30]}});
                """,
                )

        background_tasks.create(add_markers(), name="station-map-markers")


class StationTable(Component):
    name = "Station Table"
    description = (
        "All stations in the network inventory, sorted by network and station code."
    )

    async def view(self) -> None:
        state = get_tab_state()
        search = await state.run.get_search()
        stations = sorted(
            search.stations, key=lambda s: (s.network, s.station, s.location)
        )

        if not stations:
            ui.label("No stations in inventory.").classes("text-grey-6 italic")
            return

        has_depth = any(sta.depth > 0 for sta in stations)
        n_networks = len({sta.network for sta in stations})

        columns = [
            {
                "name": "network",
                "label": "Network",
                "field": "network",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "station",
                "label": "Station",
                "field": "station",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "location",
                "label": "Location",
                "field": "location",
                "sortable": True,
                "align": "left",
            },
        ]
        columns += [
            {
                "name": "lat",
                "label": "Latitude (°)",
                "field": "lat",
                "sortable": True,
                "align": "right",
            },
            {
                "name": "lon",
                "label": "Longitude (°)",
                "field": "lon",
                "sortable": True,
                "align": "right",
            },
            {
                "name": "elevation",
                "label": "Elev. (m)",
                "field": "elevation",
                "sortable": True,
                "align": "right",
            },
        ]
        if has_depth:
            columns.append(
                {
                    "name": "depth",
                    "label": "Depth (m)",
                    "field": "depth",
                    "sortable": True,
                    "align": "right",
                }
            )

        rows = []
        for sta in stations:
            row = {
                "id": sta.nsl.pretty,
                "network": sta.network,
                "station": sta.station,
                "location": sta.location or "—",
                "lat": round(sta.effective_lat, 4),
                "lon": round(sta.effective_lon, 4),
                "elevation": round(sta.elevation),
            }
            if has_depth:
                row["depth"] = round(sta.depth) if sta.depth > 0 else 0
            rows.append(row)

        # Stats strip + filter
        with ui.row().classes("items-center gap-6 mb-3 w-full"):
            filter_input = (
                ui.input(placeholder="Search stations...")
                .props('dense outlined clearable debounce="200"')
                .classes("w-64")
            )
            with filter_input.add_slot("prepend"):
                ui.icon("search").classes("text-gray-400")
            ui.space()
            with ui.row().classes("items-center gap-1"):
                ui.icon("sensors").classes("text-blue-5")
                ui.label(f"{len(stations)} stations").classes(
                    "text-subtitle2 text-grey-8"
                )
            with ui.row().classes("items-center gap-1"):
                ui.icon("hub").classes("text-teal-5")
                ui.label(f"{n_networks} networks").classes("text-subtitle2 text-grey-8")

        table = (
            ui.table(
                columns=columns,
                rows=rows,
                row_key="id",
                pagination={"rowsPerPage": 20},
            )
            .classes("w-full text-sm")
            .props("dense flat bordered")
        )
        table.add_slot(
            "body-row",
            """
            <q-tr :props="props"
                  :class="props.rowIndex % 2 === 0 ? 'bg-grey-1' : ''"
                  style="cursor: pointer">
                <q-td v-for="col in props.cols" :key="col.name" :props="props">
                    {{ col.value }}
                </q-td>
            </q-tr>
            """,
        )
        table.on(
            "row-click",
            lambda e: ui.navigate.to(f"/station/{e.args[1]['id']}"),
        )
        filter_input.bind_value_to(table, "filter")
