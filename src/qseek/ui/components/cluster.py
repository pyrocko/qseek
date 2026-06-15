from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.plotly import Plotly

from qseek.ui.analysis.cluster import _NOISE_COLOR, labels_to_colors
from qseek.ui.analysis.magnitudes import calculate_dmag_bpositive
from qseek.ui.base import Component
from qseek.ui.models import EventMinimal
from qseek.ui.utils import attach_plotly_events

_DELTA_MC = 0.5
_MIN_MAG_DIFF = 10


class ClusterAnalysis(Component):
    title = "Cluster b-Positive Analysis"
    description = """
b-positive value (left axis) and number of events (right axis) per cluster.
Noise events (label -1) are excluded. Clusters with fewer than 10 positive
magnitude differences are shown without a b-value.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis={"title": "Cluster", "type": "category"},
            yaxis={"title": "b-positive"},
            yaxis2={
                "title": "Number of Events",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            legend={
                "x": 0.01,
                "y": 0.99,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
            },
        )
        self.plot = ui.plotly(fig).classes("w-full h-64")
        self.figure = fig

    async def plot_clusters(
        self,
        events: list[EventMinimal],
        labels: np.ndarray,
    ) -> None:
        fig = self.figure

        unique_labels = sorted(set(labels.tolist()) - {-1})
        if not unique_labels:
            return

        cluster_ids: list[str] = []
        b_pos_values: list[float] = []
        b_pos_errors: list[float] = []
        n_events: list[int] = []
        colors: list[str] = []

        events_arr = np.asarray(events, dtype=object)
        for label in unique_labels:
            mask = labels == label
            cluster_events: list[EventMinimal] = events_arr[mask].tolist()

            cluster_ids.append(f"#{label}")
            colors.append(labels_to_colors([label])[0])
            n_events.append(int(mask.sum()))

            filtered = [
                ev
                for ev in cluster_events
                if ev.magnitude is not None
                and ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
            ]
            if len(filtered) < 5:
                b_pos_values.append(np.nan)
                b_pos_errors.append(np.nan)
                continue

            magnitudes = np.asarray([ev.magnitude.average for ev in filtered])
            times = np.asarray([ev.time.timestamp() for ev in filtered])
            _, mag_diff = calculate_dmag_bpositive(times, magnitudes, d_mc=_DELTA_MC)
            if len(mag_diff) < _MIN_MAG_DIFF:
                b_pos_values.append(np.nan)
                b_pos_errors.append(np.nan)
                continue

            b_pos = 1.0 / (np.log(10.0) * np.mean(mag_diff))
            b_pos_errors.append(b_pos / np.sqrt(mag_diff.size))
            b_pos_values.append(b_pos)

        fig.data = []
        fig.add_trace(
            go.Bar(
                x=cluster_ids,
                y=n_events,
                name="N Events",
                marker={"color": colors, "opacity": 0.35, "line": {"width": 0}},
                yaxis="y2",
                hoverinfo="none",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cluster_ids,
                y=b_pos_values,
                mode="markers",
                name="b-positive",
                error_y={
                    "type": "data",
                    "array": b_pos_errors,
                    "visible": True,
                    "color": "rgba(0,0,0,0.5)",
                    "thickness": 1.5,
                    "width": 4,
                },
                marker={
                    "color": colors,
                    "size": 10,
                    "line": {"color": "black", "width": 1},
                },
                yaxis="y",
                hoverinfo="none",
            )
        )
        self.plot.update()


class MagnitudeRateCluster(Component):
    title = "Magnitude Rate by Cluster"
    description = """
Magnitude of detected events over time, coloured by cluster. Toggle individual
clusters on and off via the legend. Noise events (label -1) are shown separately.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Magnitude",
            legend={
                "x": 0.01,
                "y": 0.99,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
            },
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)
        self.plot = plot
        self.figure = fig

    async def plot_events(
        self,
        events: list[EventMinimal],
        labels: np.ndarray,
        show_semblance: bool = False,
    ) -> None:
        fig = self.figure
        events_arr = np.asarray(events, dtype=object)

        def _value(ev: EventMinimal) -> float | None:
            if show_semblance:
                return ev.semblance
            if ev.magnitude is not None and ev.magnitude.average is not None:
                v = ev.magnitude.average
                return float(v) if np.isfinite(v) else None
            return None

        # Global min/max for consistent marker sizing across all clusters.
        all_values = np.asarray(
            [v for ev in events if (v := _value(ev)) is not None], dtype=float
        )
        if len(all_values) == 0:
            return
        global_min = all_values.min()
        global_max = all_values.max()

        def _sizes(values: np.ndarray) -> np.ndarray | int:
            if global_max != global_min:
                return (values - global_min) / (global_max - global_min) * 15
            return 10

        unique_labels = sorted(set(labels.tolist()))
        ordered = [lbl for lbl in unique_labels if lbl != -1]
        if -1 in unique_labels:
            ordered.append(-1)

        fig.data = []
        fig.update_layout(yaxis_title="Semblance" if show_semblance else "Magnitude")

        for label in ordered:
            mask = labels == label
            cluster_events: list[EventMinimal] = events_arr[mask].tolist()

            rows = [
                (ev.time, str(ev.uid), v)
                for ev in cluster_events
                if (v := _value(ev)) is not None
            ]
            if not rows:
                continue
            times, uids, values = zip(*rows, strict=True)
            values = np.asarray(values, dtype=float)

            if label == -1:
                color = _NOISE_COLOR
                name = "Noise"
                opacity = 0.15
            else:
                color = labels_to_colors([label])[0]
                name = f"Cluster #{label}"
                opacity = 0.4

            fig.add_trace(
                go.Scattergl(
                    x=list(times),
                    y=values,
                    mode="markers",
                    name=name,
                    customdata=list(uids),
                    marker={
                        "color": color,
                        "size": _sizes(values),
                        "line": {"width": 0},
                        "opacity": opacity,
                    },
                    hoverinfo="none",
                    hovertemplate=None,
                )
            )

        self.plot.update()
