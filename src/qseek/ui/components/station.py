from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

from qseek.ui.base import Component
from qseek.ui.utils import attach_plotly_navigate
from qseek.utils import _NSL


def _station_label(value: object) -> str:
    if hasattr(value, "pretty_str"):
        return value.pretty_str(strip=True)
    if hasattr(value, "pretty"):
        return str(value.pretty).rstrip(".")
    if isinstance(value, (list, tuple)):
        return ".".join(str(v) for v in value).rstrip(".")
    return str(value).rstrip(".")


@dataclass(slots=True)
class StationPageData:
    station_label: str
    event_rows: list[dict]
    pick_rows: list[dict]
    station_mag_rows: list[dict]


class StationComponent(Component):
    def __init__(self, data: StationPageData) -> None:
        self.data = data

    async def render(self) -> None:
        with ui.card().classes("w-[calc(50%-0.5rem)] flex-wrap shadow-2"):
            with ui.row().classes("text-h5"):
                if self.icon:
                    ui.icon(self.icon)
                ui.label(self.name)
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()


def collect_station_page_data(catalog, station_nsl: _NSL) -> StationPageData:
    station_label = station_nsl.pretty_str(strip=True)

    event_rows: list[dict] = []
    pick_rows: list[dict] = []
    station_mag_rows: list[dict] = []

    all_dists = []
    all_mags = []
    for ev in catalog.events:
        if ev.magnitude is None or not ev.magnitude.station_magnitudes:
            continue
        for sm in ev.magnitude.station_magnitudes:
            dist_km = sm.distance_epi / 1000.0
            if dist_km > 0 and np.isfinite(dist_km) and np.isfinite(sm.magnitude):
                all_dists.append(dist_km)
                all_mags.append(sm.magnitude)

    attenuation_fit: tuple[float, float] | None = None
    if len(all_dists) >= 10:
        matrix = np.vstack([np.log10(all_dists), np.ones(len(all_dists))]).T
        slope, intercept = np.linalg.lstsq(matrix, all_mags, rcond=None)[0]
        attenuation_fit = (float(slope), float(intercept))

    for ev in catalog.events:
        try:
            receiver = ev.event.receivers.get_by_nsl(station_nsl)
        except KeyError:
            continue

        if receiver.phase_arrivals:
            start_time, end_time = receiver.get_arrivals_time_window()
            event_rows.append(
                {
                    "uid": str(ev.uid),
                    "time": ev.time,
                    "start": start_time,
                    "end": end_time,
                    "n_picks": receiver.n_picks(),
                }
            )

        for phase, arrival in receiver.phase_arrivals.items():
            if arrival.traveltime_delay is None:
                continue
            pick_rows.append(
                {
                    "uid": str(ev.uid),
                    "time": ev.time,
                    "phase": phase,
                    "residual": arrival.traveltime_delay.total_seconds(),
                }
            )

        if ev.magnitude is None:
            continue

        for sm in ev.magnitude.station_magnitudes:
            if _station_label(sm.station) != station_label:
                continue

            dist_km = sm.distance_epi / 1000.0
            residual = np.nan
            if (
                attenuation_fit is not None
                and dist_km > 0
                and np.isfinite(dist_km)
                and np.isfinite(sm.magnitude)
            ):
                residual = sm.magnitude - (
                    attenuation_fit[0] * np.log10(dist_km) + attenuation_fit[1]
                )
            elif (
                ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
                and np.isfinite(sm.magnitude)
            ):
                residual = sm.magnitude - ev.magnitude.average

            station_mag_rows.append(
                {
                    "uid": str(ev.uid),
                    "time": ev.time,
                    "distance": dist_km,
                    "magnitude": sm.magnitude,
                    "snr": sm.snr,
                    "residual": residual,
                }
            )

    return StationPageData(
        station_label=station_label,
        event_rows=event_rows,
        pick_rows=pick_rows,
        station_mag_rows=station_mag_rows,
    )


class StationActiveTimeRangesPlot(StationComponent):
    name = "Station Active Time Ranges"
    description = (
        "Arrival windows at this station for each event in the filtered catalog."
    )

    async def render(self) -> None:
        with ui.card().classes("w-full flex-wrap shadow-2"):
            with ui.row().classes("text-h5"):
                if self.icon:
                    ui.icon(self.icon)
                ui.label(self.name)
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()

    async def view(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Event Index",
        )

        sorted_events = sorted(self.data.event_rows, key=lambda row: row["time"])
        x_values = []
        y_values = []
        custom_data = []

        for idx, row in enumerate(sorted_events):
            x_values.extend([row["start"], row["end"], None])
            y_values.extend([idx, idx, None])
            custom_data.extend([row["uid"], row["uid"], None])

        fig.add_trace(
            go.Scattergl(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                line={"width": 6, "color": "#5C8FA3"},
                marker={"size": 6, "color": "#3E5C68"},
                customdata=custom_data,
                hoverinfo="skip",
                hovertemplate=None,
            )
        )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)


class StationPickResidualHistogramPlot(StationComponent):
    name = "Pick Residual Histogram"
    description = "Travel-time residuals by phase for this station."

    async def view(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Residual (s)",
            yaxis_title="Count",
            barmode="overlay",
        )

        if self.data.pick_rows:
            phase_values = sorted({row["phase"] for row in self.data.pick_rows})
            for phase in phase_values:
                values = [
                    row["residual"]
                    for row in self.data.pick_rows
                    if row["phase"] == phase and np.isfinite(row["residual"])
                ]
                if values:
                    fig.add_trace(
                        go.Histogram(
                            x=values,
                            name=phase,
                            opacity=0.55,
                            nbinsx=max(10, int(np.sqrt(len(values)) * 2)),
                        )
                    )

        ui.plotly(fig).classes("w-full h-64")


class StationSNRTimePlot(StationComponent):
    name = "SNR Over Time"
    description = "Station magnitude SNR values over event time (if available)."

    async def view(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="SNR",
        )

        snr_rows = [
            row
            for row in self.data.station_mag_rows
            if np.isfinite(row["snr"]) and row["snr"] > 0
        ]
        if snr_rows:
            snr_rows = sorted(snr_rows, key=lambda row: row["time"])
            fig.add_trace(
                go.Scattergl(
                    x=[row["time"] for row in snr_rows],
                    y=[row["snr"] for row in snr_rows],
                    mode="markers",
                    line={"color": "rgba(62,92,104,0.45)", "width": 1.5},
                    marker={
                        "size": 8,
                        "color": [row["snr"] for row in snr_rows],
                        "colorscale": "Viridis",
                        "showscale": False,
                        "opacity": 0.7,
                    },
                    customdata=[row["uid"] for row in snr_rows],
                    hovertemplate="SNR: %{y:.2f}<extra></extra>",
                )
            )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)


class StationMagnitudeResidualHistogramPlot(StationComponent):
    name = "Magnitude Residual Histogram"
    description = "Distance-corrected station magnitude residual distribution."

    async def view(self) -> None:
        residual_values = [
            row["residual"]
            for row in self.data.station_mag_rows
            if np.isfinite(row["residual"])
        ]

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Residual",
            yaxis_title="Count",
        )
        if residual_values:
            fig.add_trace(
                go.Histogram(
                    x=residual_values,
                    marker={"color": "#5C8FA3"},
                    nbinsx=max(10, int(np.sqrt(len(residual_values)) * 2)),
                )
            )

        ui.plotly(fig).classes("w-full h-64")


class StationMagnitudeResidualTimePlot(StationComponent):
    name = "Magnitude Residual Over Time"
    description = "Distance-corrected residual at this station over time."

    async def view(self) -> None:
        residual_rows = [
            row for row in self.data.station_mag_rows if np.isfinite(row["residual"])
        ]
        residual_rows = sorted(residual_rows, key=lambda row: row["time"])

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Residual",
        )
        if residual_rows:
            fig.add_hline(y=0.0, line={"width": 1, "dash": "dot", "color": "gray"})
            values = [row["residual"] for row in residual_rows]
            color_limit = max(abs(float(np.min(values))), abs(float(np.max(values))))
            fig.add_trace(
                go.Scattergl(
                    x=[row["time"] for row in residual_rows],
                    y=values,
                    mode="markers",
                    line={"color": "rgba(62,92,104,0.35)", "width": 1.5},
                    marker={
                        "size": 8,
                        "color": values,
                        "colorscale": "RdBu_r",
                        "cmin": -color_limit,
                        "cmax": color_limit,
                        "showscale": False,
                        "opacity": 0.8,
                    },
                    customdata=[row["uid"] for row in residual_rows],
                    hovertemplate="Residual: %{y:.2f}<extra></extra>",
                )
            )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)
