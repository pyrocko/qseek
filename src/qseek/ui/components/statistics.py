from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from scipy.stats import gaussian_kde

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import attach_plotly_navigate


class EventRateSemblance(Component):
    name = "Event Rate"
    icon = ""
    description = """
Semblance of detected events over time. Color corresponds to event density, size of
markers corresponds to semblance value.
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
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
                        "colorscale": "viridis",
                        "showscale": False,
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
                    line={"color": "rgba(0,0,0,0.7)", "dash": "solid", "width": 3},
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
    description = """
Plot distance to octree center over time to visualize event migration. Size of markers
corresponds to magnitude and color corresponds to depth (light = shallow, dark = deep).
"""

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
    name = "Depth Sections"
    description = """
Depth of detected events along profile line through the center.
Color corresponds to time and size corresponds to semblance.
"""

    async def view(self, direction: str = "north-south") -> None:
        state = get_tab_state()
        fig = go.Figure()

        fig.update_layout(
            margin={"l": 0, "r": 55, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Distance to Center (km)",
            yaxis_title="Depth (m)",
        )

        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()

            if direction == "north-south":
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
                        "colorscale": "Turbo",
                        "showscale": True,
                        "colorbar": {
                            "title": {
                                "text": "Days",
                                "font": {"size": 10},
                            },
                            "thickness": 8,
                            "len": 0.8,
                            "outlinewidth": 0,
                            "tickfont": {"size": 9},
                            "tickformat": ".0f",
                        },
                        "size": catalog.semblances / catalog.semblances.max() * 15,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class NPicksDistribution(Component):
    name = "Picks per Event"
    description = """
Distribution of number of picks associated with detected events.
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Number of Picks",
            yaxis_title="Number of Events",
        )
        plot = ui.plotly(fig).classes("w-full h-full")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            n_picks = catalog.n_picks
            n_picks = n_picks[~np.isnan(n_picks)].astype(int)

            counts = np.bincount(n_picks)
            x = np.arange(len(counts))
            y = counts

            median = float(np.median(n_picks))

            plot.clear()
            fig.add_bar(
                x=x,
                y=y,
                name="N Picks Distribution",
                marker_color="gray",
                hoverinfo="none",
                hovertemplate=None,
            )
            fig.add_vline(
                x=median,
                line={
                    "dash": "dash",
                    "color": "rgba(0,0,0,0.4)",
                    "width": 1.5,
                },
                annotation={
                    "text": f"Median:\n{median:.0f} Picks",
                    "font": {"size": 10, "color": "rgba(0,0,0,0.5)"},
                    "xanchor": "left",
                    "yanchor": "top",
                    "showarrow": False,
                    "yref": "paper",
                    "y": 0.98,
                },
            )
            plot.update()

        background_tasks.create(update_plot())


class WadatiDiagram(Component):
    name = "Wadati Diagram"
    description = """
Showing t<sub>P</sub> vs t<sub>S</sub>-t<sub>P</sub> arrival times and the apparent
Vp/Vs ratio of detected events. The size of markers corresponds to semblance and color
corresponds to event time.
"""

    async def view(self) -> None:
        state = get_tab_state()

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="P Travel Time (s)",
            yaxis_title="S-P Travel Time (s)",
            legend={
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.75)",
            },
        )
        plot = ui.plotly(fig).classes("w-full h-full")
        attach_plotly_navigate(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()

            arrivals = defaultdict(list)
            event_times = []
            event_semblances = []
            event_uids = []

            for ev in catalog.events:
                event = ev.event
                if not event.receivers:
                    continue

                travel_time_pairs = event.receivers.get_travel_time_pairs(
                    origin_time=event.time
                )
                if len(travel_time_pairs) != 2:
                    continue

                for phase, arrival_times in travel_time_pairs.items():
                    arrivals[phase].extend(arrival_times)
                    n_travel_times = len(arrival_times)

                event_times.extend([event.time.timestamp()] * n_travel_times)
                event_semblances.extend([event.semblance] * n_travel_times)
                event_uids.extend([event.uid] * n_travel_times)

            p_key = next((ph for ph in arrivals if ph.endswith("P")), None)
            s_key = next((ph for ph in arrivals if ph.endswith("S")), None)
            if not p_key or not s_key:
                return

            p_arr = np.array(arrivals[p_key])
            s_arr = np.array(arrivals[s_key])
            sp_arr = s_arr - p_arr

            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=p_arr,
                    y=sp_arr,
                    mode="markers",
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=event_uids,
                    marker={
                        "color": np.array(event_times),
                        "colorscale": "Turbo",
                        "showscale": False,
                        "size": np.array(event_semblances) / max(event_semblances) * 15,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    showlegend=False,
                    name="Wadati Plot",
                )
            )

            mask = np.isfinite(p_arr) & np.isfinite(sp_arr) & (p_arr > 0)
            if mask.sum() > 1:
                p_clean = p_arr[mask]
                sp_clean = sp_arr[mask]
                vp_vs_median = float(np.median(sp_clean / p_clean) + 1)
                p_range = np.array([0.0, p_clean.max()])
                fig.add_trace(
                    go.Scatter(
                        x=p_range,
                        y=(vp_vs_median - 1) * p_range,
                        mode="lines",
                        name=f"Vp/Vs = {vp_vs_median:.2f}",
                        line={
                            "color": "rgba(200,50,50,0.8)",
                            "dash": "dash",
                            "width": 1.5,
                        },
                        hoverinfo="none",
                        hovertemplate=None,
                    )
                )

            plot.update()

        background_tasks.create(update_plot())
