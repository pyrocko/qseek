import asyncio
import json

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from plotly.subplots import make_subplots

from qseek.models.station import Station
from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import attach_plotly_events

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
            tip = f"<b>{station.nsl.pretty_str(strip=True)}</b><br>Elevation: {station.elevation:.0f} m"
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
    name = "Station Details"
    description = """
Essential metadata about the station.
"""

    async def view(self) -> None:
        station = self.station

        rows = [
            {"property": "NSL", "value": station.nsl.pretty_str(strip=True)},
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


class StationPickPerformance(StationComponent):
    name = "Pick Performance"
    description = (
        "Detection confidence of P and S arrivals at this station over time. "
        "P picks are shown above zero, S picks below."
    )

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Detection Value",
            showlegend=True,
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
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        nsl = self.station.nsl

        async def update_plot():
            catalog = await state.run.get_catalog()

            def collect():
                p_times, p_vals, p_uids = [], [], []
                s_times, s_vals, s_uids = [], [], []
                for ev in catalog.events:
                    uid = str(ev.uid)
                    for rcv in ev.receivers:
                        if rcv.nsl != nsl:
                            continue
                        try:
                            arrival = rcv.get_arrival("P")
                            if arrival.observed is not None:
                                p_times.append(ev.time)
                                p_vals.append(arrival.observed.detection_value)
                                p_uids.append(uid)
                        except KeyError:
                            pass
                        try:
                            arrival = rcv.get_arrival("S")
                            if arrival.observed is not None:
                                s_times.append(ev.time)
                                s_vals.append(arrival.observed.detection_value)
                                s_uids.append(uid)
                        except KeyError:
                            pass
                return p_times, p_vals, p_uids, s_times, s_vals, s_uids

            p_times, p_vals, p_uids, s_times, s_vals, s_uids = await asyncio.to_thread(
                collect
            )

            fig.data = []
            if p_vals:
                median_p = float(np.median(p_vals))
                fig.add_trace(
                    go.Scattergl(
                        name="P",
                        x=p_times,
                        y=p_vals,
                        mode="markers",
                        marker={
                            "color": "#5C8FA3",
                            "size": np.clip(np.array(p_vals) * 12, 2, 14).tolist(),
                            "opacity": 0.6,
                            "line": {"width": 0},
                        },
                        customdata=p_uids,
                        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>P: %{y:.3f}<extra></extra>",
                    )
                )
                fig.add_hline(
                    y=median_p,
                    line={"dash": "dash", "color": "#5C8FA3", "width": 1.5},
                    annotation={
                        "text": f"P median: {median_p:.3f}",
                        "font": {"size": 10, "color": "#5C8FA3"},
                        "xanchor": "right",
                        "yanchor": "bottom",
                        "showarrow": False,
                        "x": 0.99,
                    },
                )
            if s_vals:
                median_s = float(np.median(s_vals))
                neg_s = [-v for v in s_vals]
                fig.add_trace(
                    go.Scattergl(
                        name="S",
                        x=s_times,
                        y=neg_s,
                        mode="markers",
                        marker={
                            "color": "#E07B54",
                            "size": np.clip(np.array(s_vals) * 12, 2, 14).tolist(),
                            "opacity": 0.6,
                            "line": {"width": 0},
                        },
                        customdata=s_uids,
                        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>S: %{y:.3f}<extra></extra>",
                    )
                )
                fig.add_hline(
                    y=-median_s,
                    line={"dash": "dash", "color": "#E07B54", "width": 1.5},
                    annotation={
                        "text": f"S median: {median_s:.3f}",
                        "font": {"size": 10, "color": "#E07B54"},
                        "xanchor": "right",
                        "yanchor": "top",
                        "showarrow": False,
                        "x": 0.99,
                    },
                )
            plot.update()

        state.catalog_store.updated.subscribe(update_plot)
        background_tasks.create(update_plot(), name=f"station-pick-perf-{nsl.pretty}")


class StationTraveltimeResidual(StationComponent):
    name = "Traveltime Residuals"
    description = (
        "P and S traveltime residuals (observed - modelled) at this station over time. "
        "Marker size reflects detection confidence. "
        "The dashed trend line is a confidence-weighted linear fit; "
        "a non-zero slope indicates a systematic timing drift."
    )

    async def view(self) -> None:
        state = get_tab_state()

        _zeroline = {
            "zeroline": True,
            "zerolinewidth": 1.5,
            "zerolinecolor": "rgba(0,0,0,0.25)",
        }
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
        )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            showlegend=False,
        )
        fig.update_yaxes(title_text="P residual (s)", **_zeroline, row=1, col=1)
        fig.update_yaxes(title_text="S residual (s)", **_zeroline, row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        plot = ui.plotly(fig).classes("w-full h-80")
        attach_plotly_events(plot)

        nsl = self.station.nsl

        async def update_plot():
            catalog = await state.run.get_catalog()

            def collect():
                p_times, p_vals, p_uids, p_confs = [], [], [], []
                s_times, s_vals, s_uids, s_confs = [], [], [], []
                for ev in catalog.events:
                    uid = str(ev.uid)
                    for rcv in ev.receivers:
                        if rcv.nsl != nsl:
                            continue
                        try:
                            arrival = rcv.get_arrival("P")
                            if (
                                arrival.traveltime_delay is not None
                                and arrival.observed is not None
                            ):
                                p_times.append(ev.time)
                                p_vals.append(arrival.traveltime_delay.total_seconds())
                                p_uids.append(uid)
                                p_confs.append(arrival.observed.detection_value)
                        except KeyError:
                            pass
                        try:
                            arrival = rcv.get_arrival("S")
                            if (
                                arrival.traveltime_delay is not None
                                and arrival.observed is not None
                            ):
                                s_times.append(ev.time)
                                s_vals.append(arrival.traveltime_delay.total_seconds())
                                s_uids.append(uid)
                                s_confs.append(arrival.observed.detection_value)
                        except KeyError:
                            pass
                return (
                    p_times,
                    p_vals,
                    p_uids,
                    p_confs,
                    s_times,
                    s_vals,
                    s_uids,
                    s_confs,
                )

            (
                p_times,
                p_vals,
                p_uids,
                p_confs,
                s_times,
                s_vals,
                s_uids,
                s_confs,
            ) = await asyncio.to_thread(collect)

            if not p_vals and not s_vals:
                return

            fig.data = []
            fig.layout.annotations = []

            for times, vals, uids, confs, color, phase, row in (
                (p_times, p_vals, p_uids, p_confs, "#5C8FA3", "P", 1),
                (s_times, s_vals, s_uids, s_confs, "#E07B54", "S", 2),
            ):
                if not vals:
                    continue
                arr = np.array(vals)
                conf_arr = np.array(confs)
                lo, hi = np.percentile(arr, [2, 98])
                mask = (arr >= lo) & (arr <= hi)
                t_vis = [t for t, m in zip(times, mask, strict=True) if m]
                y_vis = arr[mask]
                u_vis = [u for u, m in zip(uids, mask, strict=True) if m]
                c_vis = conf_arr[mask]

                fig.add_trace(
                    go.Scattergl(
                        name=phase,
                        x=t_vis,
                        y=y_vis.tolist(),
                        mode="markers",
                        marker={
                            "color": color,
                            "size": np.clip(c_vis * 10, 2, 12).tolist(),
                            "opacity": 0.4,
                            "line": {"width": 0},
                        },
                        customdata=u_vis,
                        hovertemplate=(
                            f"%{{x|%Y-%m-%d %H:%M:%S}}<br>{phase}: %{{y:.3f}} s<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )

                if len(t_vis) >= 2:
                    t_num = np.array([t.timestamp() for t in t_vis])
                    slope, intercept = np.polyfit(t_num, y_vis, 1, w=c_vis)
                    fit_y = slope * t_num + intercept
                    slope_ms_per_day = slope * 86_400 * 1_000
                    fig.add_trace(
                        go.Scattergl(
                            name=f"{phase} trend",
                            x=t_vis,
                            y=fit_y.tolist(),
                            mode="lines",
                            line={"color": color, "width": 1.5, "dash": "longdash"},
                            hovertemplate=(
                                f"{phase} trend: {slope_ms_per_day:+.2f} ms/day<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=1,
                    )
                    fig.add_annotation(
                        x=t_vis[-1],
                        y=float(fit_y[-1]),
                        text=f"{slope_ms_per_day:+.2f} ms/day",
                        font={"size": 10, "color": color},
                        showarrow=False,
                        xanchor="right",
                        yanchor="bottom",
                        xref=f"x{row if row > 1 else ''}",
                        yref=f"y{row if row > 1 else ''}",
                    )

                fig.update_yaxes(range=[lo, hi], row=row, col=1)

            plot.update()

        state.catalog_store.updated.subscribe(update_plot)
        background_tasks.create(
            update_plot(), name=f"station-traveltime-residual-{nsl.pretty}"
        )
