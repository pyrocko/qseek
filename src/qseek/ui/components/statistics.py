from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from scipy.stats import gaussian_kde

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import attach_plotly_navigate


class SemblanceRate(Component):
    name = "Semblance Rate"
    description = """
Semblance of detected events over time. Size of markers corresponds
to semblance value.
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 80, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Semblance",
            legend={
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.75)",
            },
            yaxis2={
                "title": "Cumulative Semblance",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            records = [
                (ev.time, ev.uid, ev.semblance)
                for ev in catalog.events
                if ev.semblance is not None and np.isfinite(ev.semblance)
            ]
            if not records:
                return
            times, uids, semblances = map(np.asarray, zip(*records, strict=True))
            semblances = np.asarray(semblances, dtype=float)

            time_order = np.argsort(times)
            times_sorted = times[time_order]
            semblances_sorted = semblances[time_order]

            point_density = None
            try:
                time_numeric = np.asarray(
                    [time.timestamp() for time in times],
                    dtype=float,
                )
                scott_kde = gaussian_kde(time_numeric, bw_method="scott")
                kde = gaussian_kde(time_numeric, bw_method=scott_kde.factor * 0.1)
                point_density = kde(time_numeric)
            except (ValueError, np.linalg.LinAlgError):
                point_density = None

            scatter_times = times
            scatter_semblances = semblances
            scatter_uids = uids
            if point_density is not None:
                density_order = np.argsort(point_density)
                scatter_times = scatter_times[density_order]
                scatter_semblances = scatter_semblances[density_order]
                scatter_uids = scatter_uids[density_order]
                point_density = point_density[density_order]

            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=scatter_times,
                    y=scatter_semblances,
                    mode="markers",
                    name="Event Semblance",
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=scatter_uids,
                    marker={
                        "color": point_density
                        if point_density is not None
                        else "black",
                        "colorscale": "inferno",
                        "showscale": point_density is not None,
                        "colorbar": {
                            "title": "Density",
                            "x": 1.1,
                            "xanchor": "left",
                        },
                        "size": scatter_semblances / scatter_semblances.max() * 15,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    # hovertemplate="Time: %{x}<br>Semblance: %{y:.3f}<extra></extra>",
                )
            )
            cumulative_semblance = np.cumsum(semblances_sorted)
            fig.add_trace(
                go.Scattergl(
                    x=times_sorted,
                    y=cumulative_semblance,
                    mode="lines",
                    name="Cumulative Semblance",
                    hoverinfo="none",
                    hovertemplate=None,
                    line={"color": "red", "dash": "solid", "width": 3},
                    yaxis="y2",
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class StationCoverage(Component):
    name = "Station Coverage"
    description = """Number of stations contributing to each detected event."""

    async def view(self) -> None: ...


class MigrationPlot(Component):
    name = "Migration Plot"
    description = """Plot distance to center over time to visualize event migration. Size of markers corresponds to magnitude and color corresponds to depth."""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Distance to Center (km)",
        )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        catalog = await state.get_filtered_catalog()

        async def update_plot():
            distances = np.sqrt(
                catalog.north_shifts**2 + catalog.east_shifts**2 + catalog.depths**2
            )

            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=catalog.times,
                    y=distances,
                    mode="markers",
                    name="Migration Plot",
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=catalog.uids,
                    marker={
                        "color": catalog.depths,
                        "colorscale": "Magma",
                        "size": catalog.semblances / catalog.semblances.max() * 15,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                )
            )
            plot.update()

        catalog.updated.subscribe(update_plot)
        background_tasks.create(update_plot())


class DepthSection(Component):
    @property
    def name(self) -> str:
        axis = "North-South" if self.direction == "north-south" else "East-West"
        return f"Depth Section ({axis})"

    description = """Depth of detected events along profile line through the center. Color corresponds to time and size corresponds to magnitude."""

    def __init__(self, direction: str = "north-south") -> None:
        self.direction = direction

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()

        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Distance to Center (km)",
            yaxis_title="Depth (m)",
        )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()

            if self.direction == "north-south":
                distances = catalog.north_shifts
            else:
                distances = catalog.east_shifts

            times = catalog.times
            times_num = np.array([(t - times[0]).total_seconds() for t in times]) / (
                3600 * 24
            )
            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=distances,
                    y=-catalog.depths,
                    mode="markers",
                    name=self.name,
                    customdata=catalog.uids,
                    hoverinfo="none",
                    hovertemplate=None,
                    marker={
                        "color": times_num,
                        "colorscale": "Jet",
                        "size": catalog.semblances / catalog.semblances.max() * 15,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class NPicksDistribution(Component):
    name = "N Picks Distribution"
    description = """Distribution of number of picks associated with detected events."""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Number of Picks",
            yaxis_title="Count",
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            n_picks = catalog.n_picks
            n_picks = n_picks[~np.isnan(n_picks)].astype(int)

            counts = np.bincount(n_picks)
            x = np.arange(len(counts))
            y = counts

            plot.clear()
            fig.add_bar(
                x=x,
                y=y,
                name="N Picks Distribution",
                marker_color="gray",
                hoverinfo="none",
                hovertemplate=None,
            )
            plot.update()

        background_tasks.create(update_plot())


class StationActivityOverview(Component):
    name = "Station Activity Overview"
    description = "Active arrival windows for all stations across filtered events."

    async def render(self) -> None:
        with ui.card().classes("w-full flex-wrap shadow-2"):
            with ui.row().classes("text-h5"):
                ui.label(self.name)
            ui.label(self.description).classes("text-body2 mb-2")
            await self.view()

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Station",
        )

        plot = ui.plotly(fig).classes("w-full h-160")
        attach_plotly_navigate(plot, route="station")

        async def update_plot():
            catalog = await state.get_filtered_catalog()

            rows = []
            for ev in catalog.events:
                for receiver in ev.event.receivers:
                    if not receiver.phase_arrivals:
                        continue
                    start_time, _ = receiver.get_arrivals_time_window()
                    rows.append(
                        {
                            "uid": str(ev.uid),
                            "station": receiver.nsl.pretty_str(strip=True),
                            "start": start_time,
                        }
                    )

            if not rows:
                return

            stations = sorted({row["station"] for row in rows})
            station_to_index = {name: idx for idx, name in enumerate(stations)}

            start_timestamps = np.asarray(
                [row["start"].timestamp() for row in rows], dtype=float
            )
            tmin = float(np.min(start_timestamps))
            tmax = float(np.max(start_timestamps))

            # Bin activity in fixed 1-day windows.
            one_day = 24.0 * 3600.0
            if tmax <= tmin:
                time_edges = np.array(
                    [tmin - one_day / 2, tmax + one_day / 2], dtype=float
                )
            else:
                time_edges = np.arange(tmin, tmax + one_day, one_day, dtype=float)
                if len(time_edges) < 2:
                    time_edges = np.array([tmin, tmin + one_day], dtype=float)

            activity = np.zeros((len(stations), len(time_edges) - 1), dtype=np.int32)
            station_times = {station: [] for station in stations}
            for row in rows:
                station_times[row["station"]].append(row["start"].timestamp())

            for station, times in station_times.items():
                idx = station_to_index[station]
                activity[idx], _ = np.histogram(
                    np.asarray(times, dtype=float), time_edges
                )
                activity[activity > 0] = 1
            time_centers = [
                datetime.fromtimestamp(float(ts), tz=timezone.utc)
                for ts in 0.5 * (time_edges[:-1] + time_edges[1:])
            ]
            station_customdata = np.tile(
                np.asarray(stations, dtype=object).reshape(-1, 1),
                (1, activity.shape[1]),
            )

            plot.clear()
            fig.add_trace(
                go.Heatmap(
                    x=time_centers,
                    y=np.arange(len(stations)),
                    z=activity,
                    colorscale=[
                        [0.0, "rgba(0,0,0,0)"],  # keine Aktivität → unsichtbar
                        [1.0, "gray"],  # Aktivität → rot
                    ],
                    zmin=0,
                    customdata=station_customdata,
                    hoverinfo="none",
                    hovertemplate=None,
                    showscale=False,
                    opacity=0.5,
                )
            )

            separator_lines = [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "yref": "y",
                    "y0": idx + 0.5,
                    "y1": idx + 0.5,
                    "line": {"color": "rgba(90,90,90,0.55)", "width": 1.2},
                    "layer": "above",
                }
                for idx in range(len(stations) - 1)
            ]

            tick_vals = list(range(len(stations)))
            tick_text = stations
            if len(stations) > 80:
                step = int(np.ceil(len(stations) / 80))
                tick_vals = tick_vals[::step]
                tick_text = tick_text[::step]

            fig.update_layout(shapes=separator_lines)
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[-0.5, len(stations) - 0.5],
            )
            plot.update()

        background_tasks.create(update_plot())


class NStationsOverTime(Component):
    name = "N Stations over time"
    description = (
        """Number of stations contributing to each detected event over time."""
    )

    async def view(self) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()

        rows = []
        for ev in catalog.events:
            for receiver in ev.event.receivers:
                if not receiver.phase_arrivals:
                    continue
                start_time, _ = receiver.get_arrivals_time_window()
                rows.append(
                    {
                        "uid": str(ev.uid),
                        "station": receiver.nsl.pretty_str(strip=True),
                        "start": start_time,
                    }
                )

        if not rows:
            return

        times = np.asarray([row["start"].timestamp() for row in rows], dtype=float)
        time_edges = np.arange(
            times.min(),
            times.max() + 24 * 7 * 3600,
            24 * 7 * 3600,
            dtype=float,
        )
        station_counts = np.zeros(len(time_edges) - 1, dtype=np.int32)
        for station in {row["station"] for row in rows}:
            station_times = np.asarray(
                [row["start"].timestamp() for row in rows if row["station"] == station],
                dtype=float,
            )
            counts, _ = np.histogram(station_times, time_edges)
            station_counts += (counts > 0).astype(np.int32)

        time_centers = [
            datetime.fromtimestamp(float(ts), tz=timezone.utc)
            for ts in 0.5 * (time_edges[:-1] + time_edges[1:])
        ]

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Number of Active Stations",
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        async def update_plot():
            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=time_centers,
                    y=station_counts,
                    mode="lines",
                    name="Active Stations",
                    hoverinfo="none",
                    hovertemplate=None,
                    line={"color": "gray", "dash": "solid", "width": 3},
                )
            )
            plot.update()

        background_tasks.create(update_plot())
