import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from plotly.subplots import make_subplots

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.utils import NSL

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
                    "label": sta.nsl.pretty_str(strip=True),
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
                pagination={"rowsPerPage": 15},
            )
            .classes("w-full text-sm")
            .props("dense flat bordered")
        )
        table.on(
            "row-click",
            lambda e: ui.navigate.to(f"/station/{e.args[1]['id']}"),
        )
        filter_input.bind_value_to(table, "filter")


class StationCoverage(Component):
    name = "Station Coverage"
    description = """Number of stations contributing to each detected event."""

    async def view(self) -> None:
        state = get_tab_state()
        catalog = await state.run.get_catalog()

        if not catalog.events:
            ui.label("No events detected yet.").classes("text-grey-6 italic")
            return

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Number of Stations",
            showlegend=False,
            yaxis={"rangemode": "tozero"},
        )
        plot = ui.plotly(fig).classes("w-full h-64")

        async def update_plot():
            events = sorted(catalog.events, key=lambda e: e.time)
            n_stations = np.array([ev.receivers.n_receivers for ev in events])
            times = [ev.time for ev in events]
            median_n = float(np.median(n_stations))

            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=times,
                    y=n_stations,
                    mode="markers",
                    hoverinfo="none",
                    hovertemplate=None,
                    marker={
                        "color": "#5C8FA3",
                        "size": 4,
                        "opacity": 0.5,
                        "line": {"width": 0},
                    },
                )
            )
            fig.add_hline(
                y=median_n,
                line={
                    "dash": "dash",
                    "color": "rgba(0,0,0,0.35)",
                    "width": 1.5,
                },
                annotation={
                    "text": f"Median: {median_n:.0f}",
                    "font": {"size": 10, "color": "rgba(0,0,0,0.5)"},
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "showarrow": False,
                    "x": 0.99,
                },
            )
            plot.update()

        background_tasks.create(update_plot(), name="station-coverage-plot")


class StationsPickPerformance(Component):
    name = "Station Performance"
    description = """
Ranking of stations by cumulative detection confidence of associated P and S picks.
Stations are sorted by total confidence (P + S) to highlight the best-observed stations.
This median metric also serves as a proxy for the Signal-to-Noise ratio.
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Station",
            yaxis_title="Cumulative Detection Value",
            yaxis2={
                "title": "Median Confidence",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "zeroline": False,
                "rangemode": "tozero",
            },
            showlegend=True,
            barmode="relative",
            yaxis={
                "zeroline": True,
                "zerolinewidth": 2,
                "zerolinecolor": "rgba(0,0,0,0.25)",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        plot = ui.plotly(fig).classes("w-full h-72")
        plot.on(
            "plotly_click",
            lambda e: ui.navigate.to(f"/station/{e.args['points'][0]['x']}"),
        )

        @dataclass
        class StationStats:
            picks_p: list[float] = field(default_factory=list)
            picks_s: list[float] = field(default_factory=list)

            def total_picks_p(self) -> float:
                return sum(self.picks_p)

            def total_picks_s(self) -> float:
                return sum(self.picks_s)

            def median_p(self) -> float:
                return float(np.median(self.picks_p)) if self.picks_p else 0.0

            def median_s(self) -> float:
                return float(np.median(self.picks_s)) if self.picks_s else 0.0

            def median_confidence(self) -> float:
                return (self.median_p() + self.median_s()) / 2

        async def update_plot():
            catalog = await state.run.get_catalog()
            station_counts: dict[NSL, StationStats] = defaultdict(StationStats)

            for ev in catalog.events:
                for rcv in ev.receivers:
                    try:
                        arrival = rcv.get_arrival("P")
                        if arrival.observed is not None:
                            station_counts[rcv.nsl].picks_p.append(
                                arrival.observed.detection_value
                            )
                    except KeyError:
                        ...
                    try:
                        arrival = rcv.get_arrival("S")
                        if arrival.observed is not None:
                            station_counts[rcv.nsl].picks_s.append(
                                arrival.observed.detection_value
                            )
                    except KeyError:
                        ...

            sorted_stations = sorted(
                station_counts.items(),
                key=lambda x: x[1].median_confidence(),
                reverse=True,
            )

            labels = [nsl.pretty_str(strip=True) for nsl, _ in sorted_stations]
            p_vals = [stats.total_picks_p() for _, stats in sorted_stations]
            s_vals = [-stats.total_picks_s() for _, stats in sorted_stations]
            med_p_vals = [stats.median_p() for _, stats in sorted_stations]
            med_s_vals = [-stats.median_s() for _, stats in sorted_stations]
            med_max = max(
                max(med_p_vals, default=0),
                max((-v for v in med_s_vals), default=0),
            )

            fig.data = []
            fig.update_layout(yaxis2={"range": [-med_max, med_max]})
            fig.add_trace(
                go.Bar(
                    name="P picks",
                    x=labels,
                    y=p_vals,
                    marker_color="#5C8FA3",
                    hovertemplate="%{x}<br>P confidence: %{y:.1f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="S picks",
                    x=labels,
                    y=s_vals,
                    marker_color="#E07B54",
                    customdata=[-v for v in s_vals],
                    hovertemplate="%{x}<br>S confidence: %{customdata:.1f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="P median",
                    x=labels,
                    y=med_p_vals,
                    yaxis="y2",
                    marker_color="rgba(80,80,80,0.18)",
                    marker_line_width=0,
                    hovertemplate="%{x}<br>P median: %{y:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="S median",
                    x=labels,
                    y=med_s_vals,
                    yaxis="y2",
                    marker_color="rgba(80,80,80,0.18)",
                    marker_line_width=0,
                    customdata=[-v for v in med_s_vals],
                    hovertemplate="%{x}<br>S median: %{customdata:.3f}<extra></extra>",
                )
            )
            plot.update()

        background_tasks.create(update_plot(), name="station-picks-plot")


class StationTraveltimeResiduals(Component):
    name = "Station Traveltime Residuals"
    description = """
Distribution of traveltime residuals (observed &minus; modelled) per station for P and S phases,
pooled across all detected events. Narrow, zero-centred violins indicate well-constrained stations;
a shifted centre reveals a station-specific delay. Stations are sorted by median absolute residual
(best-performing left).
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.5, 0.5],
        )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        _zeroline = {
            "zeroline": True,
            "zerolinewidth": 1.5,
            "zerolinecolor": "rgba(0,0,0,0.25)",
        }
        fig.update_yaxes(title_text="P residual (s)", **_zeroline, row=1, col=1)
        fig.update_yaxes(title_text="S residual (s)", **_zeroline, row=2, col=1)
        fig.update_xaxes(title_text="Station", row=2, col=1)

        plot = ui.plotly(fig).classes("w-full h-96")
        plot.on(
            "plotly_click",
            lambda e: ui.navigate.to(f"/station/{e.args['points'][0]['x']}"),
        )

        @dataclass
        class StationDelays:
            p: list[float] = field(default_factory=list)
            s: list[float] = field(default_factory=list)

            def median_abs(self) -> float:
                combined = self.p + self.s
                return float(np.median(np.abs(combined))) if combined else float("inf")

        async def update_plot():
            catalog = await state.run.get_catalog()
            delays: dict[NSL, StationDelays] = defaultdict(StationDelays)

            for ev in catalog.events:
                for rcv in ev.receivers:
                    try:
                        arrival = rcv.get_arrival("P")
                        if arrival.traveltime_delay is not None:
                            delays[rcv.nsl].p.append(
                                arrival.traveltime_delay.total_seconds()
                            )
                    except KeyError:
                        ...
                    try:
                        arrival = rcv.get_arrival("S")
                        if arrival.traveltime_delay is not None:
                            delays[rcv.nsl].s.append(
                                arrival.traveltime_delay.total_seconds()
                            )
                    except KeyError:
                        ...

            sorted_stations = sorted(
                delays.items(),
                key=lambda x: x[1].median_abs(),
            )

            # Clip each phase independently to its own 2nd-98th percentile
            all_p = [v for d in delays.values() for v in d.p]
            all_s = [v for d in delays.values() for v in d.s]
            if not all_p and not all_s:
                return
            p_lo, p_hi = np.percentile(all_p, [2, 98]) if all_p else (0, 0)
            s_lo, s_hi = np.percentile(all_s, [2, 98]) if all_s else (0, 0)

            p_x, p_y = [], []
            s_x, s_y = [], []
            for nsl, d in sorted_stations:
                label = nsl.pretty_str(strip=True)
                clipped_p = [v for v in d.p if p_lo <= v <= p_hi]
                clipped_s = [v for v in d.s if s_lo <= v <= s_hi]
                if clipped_p:
                    p_x.extend([label] * len(clipped_p))
                    p_y.extend(clipped_p)
                if clipped_s:
                    s_x.extend([label] * len(clipped_s))
                    s_y.extend(clipped_s)

            fig.data = []
            fig.update_yaxes(range=[p_lo, p_hi], row=1, col=1)
            fig.update_yaxes(range=[s_lo, s_hi], row=2, col=1)

            if p_y:
                fig.add_trace(
                    go.Violin(
                        name="P",
                        x=p_x,
                        y=p_y,
                        line_color="#5C8FA3",
                        fillcolor="rgba(92,143,163,0.3)",
                        meanline_visible=True,
                        box_visible=True,
                        points=False,
                        scalemode="width",
                        spanmode="hard",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
            if s_y:
                fig.add_trace(
                    go.Violin(
                        name="S",
                        x=s_x,
                        y=s_y,
                        line_color="#E07B54",
                        fillcolor="rgba(224,123,84,0.3)",
                        meanline_visible=True,
                        box_visible=True,
                        points=False,
                        scalemode="width",
                        spanmode="hard",
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )
            plot.update()

        background_tasks.create(update_plot(), name="station-traveltime-residuals")
