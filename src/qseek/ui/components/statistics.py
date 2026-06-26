from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.plotly import Plotly
from scipy.stats import gaussian_kde

from qseek.ui.analysis.vpvs import PSCollection
from qseek.ui.base import Panel
from qseek.ui.models import EventMinimal
from qseek.ui.state import CatalogStore
from qseek.ui.utils import attach_plotly_events
from qseek.utils import async_weighted_median


class EventRate(Panel):
    title = "Event Rate"
    description = """
Number of detected events over time.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Events / bin",
            yaxis2={
                "title": "Cumulative Events",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            showlegend=False,
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")
        self.figure = fig

    async def add_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        if not events:
            return

        times = [ev.time for ev in events]
        duration = times[-1] - times[0]

        if duration > timedelta(days=180):
            bin_sec = int(timedelta(days=1).total_seconds())
            bin_label = "Events / day"
        elif duration > timedelta(days=14):
            bin_sec = int(timedelta(hours=12).total_seconds())
            bin_label = "Events / 12 hours"
        elif duration > timedelta(hours=72):
            bin_sec = int(timedelta(hours=6).total_seconds())
            bin_label = "Events / 6 hours"
        elif duration > timedelta(hours=24):
            bin_sec = int(timedelta(hours=1).total_seconds())
            bin_label = "Events / hour"
        elif duration > timedelta(hours=2):
            bin_sec = int(timedelta(minutes=10).total_seconds())
            bin_label = "Events / 10 min"
        else:
            bin_sec = int(timedelta(minutes=1).total_seconds())
            bin_label = "Events / minute"

        time_numeric = np.array([t.timestamp() for t in times])
        t0 = np.floor(time_numeric.min() / bin_sec) * bin_sec
        bin_edges = np.arange(t0, time_numeric.max() + bin_sec, bin_sec)
        counts, _ = np.histogram(time_numeric, bins=bin_edges)

        bin_starts = [
            datetime.fromtimestamp(e, tz=timezone.utc) for e in bin_edges[:-1]
        ]
        cumulative = np.cumsum(counts)

        fig.data = []
        fig.update_layout(yaxis_title=bin_label)
        fig.add_bar(
            x=bin_starts,
            y=counts,
            name=bin_label,
            marker={"color": "gray", "line": {"width": 0}},
            width=bin_sec * 1000,
            hoverinfo="none",
            hovertemplate=None,
            showlegend=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=bin_starts,
                y=cumulative,
                mode="lines",
                name="Cumulative",
                line={"color": "black", "width": 1.5},
                hoverinfo="none",
                hovertemplate=None,
                yaxis="y2",
            )
        )
        self.plot.update()

    def attach_catalog(self, catalog: CatalogStore) -> None:
        async def update() -> None:
            await self.add_events(catalog.events)

        catalog.new_events.subscribe(update)
        catalog.updated.subscribe(update)


class NPicksDistribution(Panel):
    title = "Number of Picks Distribution"
    description = """Distribution of the number of phase picks per detected event."""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Number of Picks",
            yaxis_title="Number of Events",
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-full")
        attach_plotly_events(self.plot)
        self.figure = fig

    async def plot_picks(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        n_picks = np.array(
            [ev.n_picks for ev in events if ev.n_picks is not None], dtype=float
        )
        n_picks = n_picks[~np.isnan(n_picks)].astype(int)

        counts = np.bincount(n_picks)
        x = np.arange(len(counts))
        y = counts
        median = float(np.median(n_picks))

        fig.data = []
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
        self.plot.update()


class SemblanceDistribution(Panel):
    title = "Semblance Distribution"
    description = """Distribution of semblance values across all detected events."""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Semblance",
            yaxis_title="Number of Events",
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-full")
        self.figure = fig

    async def plot_distribution(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        semblances = np.array([ev.semblance for ev in events], dtype=float)
        semblances = semblances[np.isfinite(semblances)]
        if not semblances.size:
            return

        counts, edges = np.histogram(semblances, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2
        median = float(np.median(semblances))

        fig.data = []
        fig.add_bar(
            x=centers,
            y=counts,
            name="Semblance Distribution",
            marker_color="gray",
            hoverinfo="none",
            hovertemplate=None,
        )
        fig.add_vline(
            x=median,
            line={"dash": "dash", "color": "rgba(0,0,0,0.4)", "width": 1.5},
            annotation={
                "text": f"Median: {median:.3f}",
                "font": {"size": 10, "color": "rgba(0,0,0,0.5)"},
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "yref": "paper",
                "y": 0.98,
            },
        )
        self.plot.update()


class WadatiDiagram(Panel):
    title = "Wadati Diagram"
    description = """
P travel time vs. S-P travel time across all events. The slope gives
<i>V<sub>P</sub>/V<sub>S</sub></i>.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="P Travel Time (s)",
            yaxis_title="S-P Travel Time (s)",
            xaxis={"rangemode": "nonnegative"},
            yaxis={"scaleanchor": "x", "scaleratio": 1, "rangemode": "nonnegative"},
            legend={
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.75)",
            },
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-full")
        attach_plotly_events(self.plot)
        self.figure = fig

    async def plot_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        ps_collection = PSCollection()

        for ev in events:
            ps_collection.add_event(ev.event)

        p_arr = ps_collection.get_travel_times("P")
        s_arr = ps_collection.get_travel_times("S")
        event_uids = [tt.event.uid for tt in ps_collection.travel_times]
        pick_confidences = np.min(
            [
                ps_collection.get_confidences("P"),
                ps_collection.get_confidences("S"),
            ],
            axis=0,
        )

        sp_arr = s_arr - p_arr

        point_density = None
        n_picks = s_arr.size
        if n_picks >= 3:
            try:
                pts = np.vstack([p_arr, sp_arr])
                max_samples = 5_000
                if n_picks > max_samples:
                    rng = np.random.default_rng(0)
                    sample_idx = rng.choice(n_picks, max_samples, replace=False)
                    kde = await asyncio.to_thread(
                        gaussian_kde, pts[:, sample_idx], bw_method="scott"
                    )
                else:
                    kde = await asyncio.to_thread(gaussian_kde, pts, bw_method="scott")
                point_density = await asyncio.to_thread(kde, pts)
                order = await asyncio.to_thread(np.argsort, point_density)
                p_arr = p_arr[order]
                sp_arr = sp_arr[order]
                point_density = point_density[order]
                event_uids = [event_uids[i] for i in order]
                pick_confidences = pick_confidences[order]
            except (ValueError, np.linalg.LinAlgError):
                point_density = None

        fig.data = []
        fig.add_trace(
            go.Scattergl(
                x=p_arr,
                y=sp_arr,
                mode="markers",
                hoverinfo="none",
                hovertemplate=None,
                customdata=event_uids,
                marker={
                    "color": point_density if point_density is not None else "black",
                    "colorscale": "viridis",
                    "showscale": False,
                    "size": pick_confidences / pick_confidences.max() * 10,
                    "line": {"width": 0},
                    "opacity": 0.1,
                },
                showlegend=False,
                name="Wadati Plot",
            )
        )

        mask = np.isfinite(p_arr) & np.isfinite(sp_arr) & (p_arr > 0)
        if mask.sum() > 1:
            p_clean = p_arr[mask]
            sp_clean = sp_arr[mask]
            median = await async_weighted_median(
                sp_clean / p_clean,
                weights=pick_confidences[mask],
            )
            vp_vs_median = float(median + 1)
            p_range = np.array([0.0, p_clean.max()])
            fig.add_trace(
                go.Scattergl(
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

        self.plot.update()
