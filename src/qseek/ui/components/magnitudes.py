from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.plotly import Plotly
from scipy.stats import gaussian_kde, norm

from qseek.ui.analysis.magnitudes import (
    calculate_dmag_bpositive,
    calculate_entire_magnitude_fit,
    ogata_katsura,
    prob_ogata_katsura,
)
from qseek.ui.base import Panel
from qseek.ui.models import EventMinimal
from qseek.ui.state import CatalogStore
from qseek.ui.utils import attach_plotly_events


class MagnitudeFrequency(Panel):
    title = "Magnitude Frequency Distribution"
    description = """
Entire magnitude range (EMR) fit to the data using Ogata-Katsura (1993). Estimation of
the magnitude of completeness using maximum curvature (MaxC) and EMR fit.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Magnitude",
            yaxis={
                "title": "Probability Density",
                "type": "log",
                "exponentformat": "E",
            },
            yaxis2={
                "title": "Ogata-Katsura Probability",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            legend={"x": 1, "y": 1, "xanchor": "right", "yanchor": "top"},
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")
        self.figure = fig

    async def plot_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        magnitudes = np.asarray(
            [
                ev.magnitude.average
                for ev in events
                if ev.magnitude is not None and ev.magnitude.average is not None
            ],
            dtype=float,
        )
        if len(magnitudes) == 0:
            return
        bin_width = 0.1
        bin_edges = np.arange(
            np.min(magnitudes) - bin_width / 2,
            np.max(magnitudes) + bin_width / 2,
            bin_width,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ev_count, _ = np.histogram(magnitudes, bins=bin_edges)
        mc_max_curvature = bin_edges[np.argmax(ev_count)]

        b_value, mu, sigma = calculate_entire_magnitude_fit(magnitudes)
        b_value_sigma = b_value / np.sqrt(magnitudes.size)
        m = magnitudes
        scale_fac = len(m[((m >= bin_edges[0]) & (m <= bin_edges[-1]))]) / np.sum(
            ogata_katsura(bin_centers, b_value, mu, sigma)
        )
        ogata_katsura_curve = scale_fac * ogata_katsura(bin_centers, b_value, mu, sigma)
        ogata_katsura_prob = prob_ogata_katsura(bin_centers, b_value, mu, sigma)
        og_mc_90 = mu + sigma * norm.ppf(0.9)
        og_mc_99 = mu + sigma * norm.ppf(0.99)

        fig.data = []
        fig.add_bar(
            x=bin_edges[:-1],
            y=ev_count,
            name="Magnitude Distribution",
            marker_color="rgba(180, 185, 195, 0.4)",
            marker_line_color="rgba(150, 155, 165, 0.5)",
            marker_line_width=0.5,
            hoverinfo="none",
            hovertemplate=None,
            showlegend=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=bin_centers,
                y=ogata_katsura_curve,
                mode="lines",
                line={"color": "#E07B54", "width": 2},
                hoverinfo="none",
                name=f"b={b_value:.2f} ±{b_value_sigma:.2f}",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=bin_centers,
                y=ogata_katsura_prob,
                mode="lines",
                name="Ogata-Katsura Probability",
                yaxis="y2",
                line={"color": "#5B8DB8", "width": 2},
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.add_vline(
            x=mc_max_curvature,
            y0=0,
            line_dash="dash",
            line_color="#4CAF82",
            name=f"MaxC: {mc_max_curvature:.2f}",
            showlegend=True,
        )
        fig.add_vline(
            x=og_mc_90,
            y0=0,
            line_dash="dash",
            line_color="#5B8DB8",
            name=f"EMR (90%): {og_mc_90:.2f}",
            showlegend=True,
        )
        fig.add_vline(
            x=og_mc_99,
            y0=0,
            line_dash="dot",
            line_color="#5B8DB8",
            name=f"EMR (99%): {og_mc_99:.2f}",
            showlegend=True,
        )
        self.plot.update()


class MagnitudeRate(Panel):
    title = "Magnitude Rate"
    description = """
Magnitude of detected events over time. Size of markers corresponds to magnitude value.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(
        self, show_semblance: bool = False, show_density: bool = False
    ) -> None:
        super().__init__()
        self.show_semblance = show_semblance
        self.show_density = show_density
        self._last_cumulative_mag = 0.0
        self._scott_kde = 0.0

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Magnitude",
            showlegend=False,
            yaxis2={
                "title": "Cumulative Magnitude",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")

        attach_plotly_events(self.plot)
        self.figure = fig

    def _get_data(
        self, events: list[EventMinimal]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.show_semblance:
            events = [(ev.time, ev.uid, ev.semblance) for ev in events]
        else:
            events = [
                (ev.time, ev.uid, ev.magnitude.average)
                for ev in events
                if ev.magnitude is not None
                and ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
            ]
        if not events:
            return (), (), ()

        return map(np.asarray, zip(*events, strict=True))

    def get_density(
        self, times: np.ndarray, recalculate_scott: bool = True
    ) -> np.ndarray | None:
        try:
            time_numeric = np.asarray(
                [time.timestamp() for time in times],
                dtype=float,
            )
            if recalculate_scott or self._scott_kde == 0.0:
                self._scott_kde = gaussian_kde(time_numeric, bw_method="scott")
            kde = gaussian_kde(time_numeric, bw_method=self._scott_kde.factor * 0.1)
            return kde(time_numeric)
        except (ValueError, np.linalg.LinAlgError):
            ui.notify(
                "Could not compute point density for magnitude rate plot.",
                type="warn",
            )
            return None

    async def plot_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        times, uids, magnitudes = self._get_data(events)
        if len(times) == 0:
            return

        point_density = self.get_density(times) if self.show_density else None

        scatter_times = times
        scatter_magnitudes = magnitudes
        scatter_uids = uids

        if point_density is not None:
            density_order = np.argsort(point_density)
            scatter_times = scatter_times[density_order]
            scatter_magnitudes = scatter_magnitudes[density_order]
            scatter_uids = scatter_uids[density_order]
            point_density = point_density[density_order]

        fig.data = []
        min_mag = scatter_magnitudes.min() if len(scatter_magnitudes) > 0 else 0
        fig.add_trace(
            go.Scattergl(
                x=scatter_times,
                y=scatter_magnitudes,
                mode="markers",
                name="Event Magnitude",
                customdata=scatter_uids,
                marker={
                    "color": point_density if point_density is not None else "black",
                    "colorscale": "Cividis",
                    "showscale": False,
                    "size": (scatter_magnitudes - min_mag)
                    / (scatter_magnitudes.max() - min_mag)
                    * 15
                    if scatter_magnitudes.max() != min_mag
                    else 10,
                    "line": {"width": 0},
                    "opacity": 0.3,
                },
                hoverinfo="none",
                hovertemplate=None,
            )
        )

        # Keep a time-sorted copy for cumulative line computation.
        time_order = np.argsort(times)
        times_sorted = times[time_order]
        magnitudes_sorted = magnitudes[time_order]

        moment_magnitudes = np.power(10, 1.5 * magnitudes_sorted + 9.1)
        cumulative_magnitudes = np.cumsum(moment_magnitudes)
        self._last_cumulative_mag = cumulative_magnitudes[-1]
        fig.add_trace(
            go.Scattergl(
                x=times_sorted,
                y=cumulative_magnitudes,
                mode="lines",
                name="Cumulative Magnitude M0",
                line={
                    "color": "black",
                    "dash": "solid",
                    "width": 1.5,
                },
                hoverinfo="none",
                hovertemplate=None,
                yaxis="y2",
            )
        )
        fig.update_layout(
            uirevision=True,
            xaxis={"range": [times.min(), times.max()]},
        )

        self.plot.update()

    def attach_catalog(self, catalog: CatalogStore) -> None:
        async def update_plot() -> None:
            await self.plot_events(catalog.events)

        catalog.new_events.subscribe(update_plot)
        catalog.updated.subscribe(update_plot)


class MagnitudeFrequencyBPositive(Panel):
    title = "Magnitude Frequency b-Positive"
    description = """
Frequency of positive magnitude differences between consecutive events, which can be
used to estimate the b-value of the magnitude distribution.
"""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="ΔM",
            yaxis_title="Frequency",
            showlegend=False,
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(self.plot)
        self.figure = fig

    async def plot_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        filtered = [
            ev
            for ev in events
            if ev.magnitude is not None
            and ev.magnitude.average is not None
            and np.isfinite(ev.magnitude.average)
            and ev.magnitude.average >= 0
        ]
        magnitudes = np.asarray([ev.magnitude.average for ev in filtered])
        times = np.asarray([ev.time.timestamp() for ev in filtered])

        if len(magnitudes) == 0:
            return

        delta_mc = 0.5
        _, mag_diff = calculate_dmag_bpositive(times, magnitudes, d_mc=delta_mc)
        if len(mag_diff) == 0:
            return

        binedges = np.arange(min(mag_diff) - 0.05, max(mag_diff) + 0.05, 0.1)
        bincenters = (binedges[:-1] + binedges[1:]) / 2

        hist_vals, _ = np.histogram(mag_diff, bins=len(bincenters))
        bvalue_pos = 1.0 / (np.log(10.0) * np.mean(mag_diff))
        b_value_pos_uncertainty = bvalue_pos / np.sqrt(mag_diff.size)

        a = np.log10(np.max(hist_vals)) if np.max(hist_vals) > 0 else np.nan
        log10_n_pred = a - bvalue_pos * bincenters
        n_pred = 10 ** (log10_n_pred)

        fig.data = []
        fig.add_trace(
            go.Bar(
                x=bincenters,
                y=hist_vals,
                name="Magnitude Histogram",
                marker={
                    "color": "rgba(180, 185, 195, 0.4)",
                    "line": {"color": "rgba(150, 155, 165, 0.5)", "width": 0.5},
                },
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=bincenters,
                y=n_pred,
                mode="lines",
                name=f"b pos.={bvalue_pos:.2f} ±{b_value_pos_uncertainty:.2f}",
                line={"color": "#E07B54", "width": 2, "dash": "dash"},
                hoverinfo="none",
            )
        )
        fig.update_layout(
            yaxis_type="log",
            showlegend=True,
            legend={
                "x": 0.99,
                "y": 0.99,
                "xanchor": "right",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
            },
        )
        self.plot.update()


class MagnitudeStatisticsOverTime(Panel):
    title = "Magnitude Statistics Over Time"
    description = """
b-value (b-positive method) and magnitude of completeness (MaxC) computed in sliding
windows of 500 events, advancing 250 events at a time.
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
            yaxis={"title": "b-value (b-positive)"},
            yaxis2={
                "title": "Mc (MaxC)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            legend={"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")
        self.figure = fig

    async def plot_events(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        filtered = sorted(
            (
                ev
                for ev in events
                if ev.magnitude is not None
                and ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
            ),
            key=lambda ev: ev.time,
        )
        if len(filtered) < 500:
            return

        window_size = 500
        step = 10
        delta_mc = 0.5
        bin_width = 0.1

        center_times = []
        b_pos_values: list[float] = []
        b_pos_errors: list[float] = []
        mc_maxc_values: list[float] = []

        for start in range(0, len(filtered) - window_size + 1, step):
            win_events = filtered[start : start + window_size]
            win_times = np.asarray(
                [ev.time.timestamp() for ev in win_events],
                dtype=float,
            )
            win_mags = np.asarray(
                [ev.magnitude.average for ev in win_events],
                dtype=float,
            )

            center_times.append(win_events[window_size // 2].time)

            _, mag_diff = calculate_dmag_bpositive(win_times, win_mags, d_mc=delta_mc)
            if len(mag_diff) >= 10:
                b_pos = 1.0 / (np.log(10.0) * np.mean(mag_diff))
                b_pos_err = b_pos / np.sqrt(mag_diff.size)
            else:
                b_pos, b_pos_err = np.nan, np.nan
            b_pos_values.append(b_pos)
            b_pos_errors.append(b_pos_err)

            bin_edges = np.arange(
                np.min(win_mags) - bin_width / 2,
                np.max(win_mags) + bin_width,
                bin_width,
            )
            ev_count, _ = np.histogram(win_mags, bins=bin_edges)
            mc_maxc_values.append(bin_edges[np.argmax(ev_count)])

        if not center_times:
            return

        _color_b = "#E07B54"
        _color_b_band = "rgba(224,123,84,0.2)"
        _color_mc = "#5B8DB8"
        _color_mc_band = "rgba(91,141,184,0.2)"

        b_pos_arr = np.asarray(b_pos_values, dtype=float)
        b_pos_err_arr = np.asarray(b_pos_errors, dtype=float)
        b_upper = (b_pos_arr + b_pos_err_arr).tolist()
        b_lower = (b_pos_arr - b_pos_err_arr).tolist()

        mc_arr = np.asarray(mc_maxc_values, dtype=float)
        mc_upper = (mc_arr + bin_width / 2).tolist()
        mc_lower = (mc_arr - bin_width / 2).tolist()

        fig.data = []
        fig.update_layout(xaxis={"range": [filtered[0].time, filtered[-1].time]})

        fig.add_trace(
            go.Scatter(
                x=center_times + center_times[::-1],
                y=b_upper + b_lower[::-1],
                fill="toself",
                fillcolor=_color_b_band,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=center_times,
                y=b_pos_values,
                mode="lines",
                name="b-positive",
                line={"color": _color_b, "width": 1.5},
                marker={"color": _color_b, "size": 6, "line": {"width": 0}},
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=center_times + center_times[::-1],
                y=mc_upper + mc_lower[::-1],
                fill="toself",
                fillcolor=_color_mc_band,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=center_times,
                y=mc_maxc_values,
                mode="lines",
                name="Mc (MaxC)",
                line={"color": _color_mc, "width": 1.5},
                marker={"color": _color_mc, "size": 6, "line": {"width": 0}},
                yaxis="y2",
            )
        )
        self.plot.update()


class StationsMagnitudesResiduals(Panel):
    title = "Station Magnitude Residuals"
    description = """Distance-corrected station magnitude residuals per station."""
    plot: Plotly | None = None
    figure: go.Figure | None = None

    def __init__(self) -> None:
        super().__init__()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Station",
            yaxis_title="Magnitude Residual",
            showlegend=False,
        )
        with self:
            self.plot = ui.plotly(fig).classes("w-full h-64")
        self.figure = fig

    async def plot_residuals(self, events: list[EventMinimal]) -> None:
        fig = self.figure
        station_dict: dict[str, list[float]] = {}

        for ev in events:
            if (
                ev.magnitude is None
                or ev.magnitude.average is None
                or not ev.magnitude.station_magnitudes
            ):
                continue

            ev_mag = ev.magnitude.average
            for sm in ev.magnitude.station_magnitudes:
                res = ev_mag - sm.magnitude
                if not np.isfinite(res):
                    continue

                if hasattr(sm.station, "pretty"):
                    name = sm.station.pretty
                else:
                    name = ".".join(str(p) for p in sm.station)

                station_dict.setdefault(name, []).append(res)

        stations = [k for k, v in station_dict.items() if len(v) >= 5]
        if not stations:
            return

        stations.sort(key=lambda k: float(np.median(station_dict[k])))

        fig.data = []
        for st in stations:
            fig.add_trace(
                go.Violin(
                    y=station_dict[st],
                    name=st,
                    box={"visible": True},
                    meanline={"visible": True},
                    points="outliers",
                    marker={"size": 4, "color": "black", "opacity": 0.3},
                    line={"width": 1, "color": "black"},
                    fillcolor="rgba(46, 139, 87, 0.25)",
                    hoverinfo="none",
                    hovertemplate=None,
                )
            )

        fig.update_layout(xaxis={"type": "category"})
        self.plot.update()
