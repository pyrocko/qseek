from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import gaussian_kde, norm

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import attach_plotly_events

LOG_10 = np.log(10)


def magnitude_outlier_filer(magnitudes: np.ndarray) -> np.ndarray:
    """Filter out magnitude outliers using the z-score method."""
    if len(magnitudes) == 0:
        return magnitudes, np.array([], dtype=bool)
    mean = np.mean(magnitudes)
    std = np.std(magnitudes)
    if std == 0:
        return magnitudes, np.ones(len(magnitudes), dtype=bool)
    z_scores = (magnitudes - mean) / std
    mask = np.abs(z_scores) < 3
    return magnitudes[mask], mask


class MagnitudeFrequency(Component):
    name = "Magnitude Frequency Distribution"
    description = """
Entire magnitude range (EMR) fit to the data using Ogata-Katsura (1993).
Estimation of the magintude of completeness using maximum curvature (MaxC) and EMR fit.
    """

    def _log_likelihood_func(
        self,
        magnitude: float,
        beta: float,
        mu: float,
        sigma: float,
    ):
        log_gr = np.log(beta) - beta * (magnitude - mu) - 0.5 * beta**2 * sigma**2
        log_qm = norm.logcdf((magnitude - mu) / sigma)
        return log_gr + log_qm

    def _neg_log_likelihood_func(
        self,
        args: tuple[float, float, float],
        magnitudes: np.ndarray,
    ):
        b, mu, sigma = np.square(args)
        beta = b * LOG_10
        return -np.sum(self._log_likelihood_func(magnitudes, beta, mu, sigma))

    async def _calculate_entire_magnitude_fit(self, magnitudes: np.ndarray):
        x0 = [np.sqrt(1.0), np.min(magnitudes) + 1, np.sqrt(1.0)]
        res = minimize(
            self._neg_log_likelihood_func, x0, args=(magnitudes,), method="Powell"
        )
        sqrtb, sqrtmu, sqrtsigma = res.x
        b = np.square(sqrtb)
        mu = np.square(sqrtmu)
        sigma = np.square(sqrtsigma)
        return b, mu, sigma

    def _prob_ogata_katsura(self, mbinvalues, b, mu, sigma):
        dum = (mbinvalues - mu) / (np.sqrt(2.0) * sigma)
        return 0.5 * (1.0 + erf(dum))

    def _ogata_katsura(self, mbinvalues, b, mu, sigma):
        beta = LOG_10 * b
        dum = (mbinvalues - mu) / (np.sqrt(2.0) * sigma)
        qm = 0.5 * (1.0 + erf(dum))
        gr = beta * np.exp(
            -beta * (mbinvalues - mu) - np.square(beta) * np.square(sigma) / 2.0
        )
        return gr * qm

    async def view(self) -> None:
        state = get_tab_state()
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
        plot = ui.plotly(fig).classes("w-full h-64")

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            magnitudes = np.asarray(
                [
                    ev.magnitude.average
                    for ev in catalog.events
                    if ev.magnitude is not None and ev.magnitude.average is not None
                ],
                dtype=float,
            )
            if len(magnitudes) == 0:
                return
            # magnitudes, _ = magnitude_outlier_filer(magnitudes)
            bin_width = 0.1
            bin_edges = np.arange(
                np.min(magnitudes) - bin_width / 2,
                np.max(magnitudes) + bin_width / 2,
                bin_width,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ev_count, _ = np.histogram(magnitudes, bins=bin_edges)
            mc_max_curvature = bin_edges[np.argmax(ev_count)]

            b_value, mu, sigma = await self._calculate_entire_magnitude_fit(magnitudes)
            b_value_sigma = b_value / np.sqrt(magnitudes.size)
            m = magnitudes
            scale_fac = len(m[((m >= bin_edges[0]) & (m <= bin_edges[-1]))]) / np.sum(
                self._ogata_katsura(bin_centers, b_value, mu, sigma)
            )
            ogata_katsura = scale_fac * self._ogata_katsura(
                bin_centers, b_value, mu, sigma
            )
            ogata_katsura_prob = self._prob_ogata_katsura(
                bin_centers, b_value, mu, sigma
            )
            og_mc_90 = mu + sigma * norm.ppf(0.9)
            og_mc_99 = mu + sigma * norm.ppf(0.99)

            fig.data = []

            fig.add_bar(
                x=bin_edges[:-1],
                y=ev_count,
                name="Magnitude Distribution",
                marker_color="lightgray",
                hoverinfo="none",
                hovertemplate=None,
                showlegend=False,
            )
            fig.add_trace(
                go.Scattergl(
                    x=bin_centers,
                    y=ogata_katsura,
                    mode="lines",
                    line={"color": "red", "width": 1.5},
                    hoverinfo="none",
                    opacity=0.6,
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
                    line={"color": "blue", "width": 1.5},
                    opacity=0.6,
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_vline(
                x=mc_max_curvature,
                y0=0,
                line_dash="dash",
                line_color="darkgreen",
                name=f"MaxC: {mc_max_curvature:.2f}",
                showlegend=True,
            )
            fig.add_vline(
                x=og_mc_90,
                y0=0,
                line_dash="dash",
                line_color="blue",
                name=f"EMR (90%): {og_mc_90:.2f}",
                showlegend=True,
            )
            fig.add_vline(
                x=og_mc_99,
                y0=0,
                line_dash="dot",
                line_color="blue",
                name=f"EMR (99%): {og_mc_99:.2f}",
                showlegend=True,
            )

            plot.update()

        state.proxy_catalog.updated.subscribe(update_plot)
        background_tasks.create(update_plot())


class MagnitudeSemblance(Component):
    name = "Magnitude vs Semblance"
    description = """Magnitude of detected events over their semblance value."""

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Magnitude",
            yaxis_title="Semblance",
        )
        # fig.update_yaxes(scaleanchor="x", scaleratio=1)
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            magnitudes = np.asarray(
                [
                    ev.magnitude.average if ev.magnitude is not None else np.nan
                    for ev in catalog.events
                ],
                dtype=float,
            )
            uids = np.asarray([str(ev.uid) for ev in catalog.events])
            finite_mask = np.isfinite(magnitudes)
            magnitudes = magnitudes[finite_mask]
            semblances = np.asarray(catalog.semblances, dtype=float)[finite_mask]
            uids = uids[finite_mask]
            if len(magnitudes) == 0:
                return
            # magnitudes, mask = magnitude_outlier_filer(magnitudes)
            # semblances = semblances[mask]
            # uids = uids[mask]

            point_density = None
            if len(magnitudes) >= 3:
                try:
                    samples = np.vstack([semblances, magnitudes])
                    kde = gaussian_kde(samples, bw_method=0.1)
                    point_density = kde(samples)
                except (ValueError, np.linalg.LinAlgError):
                    point_density = None

            if point_density is not None:
                order = np.argsort(point_density)
                semblances = semblances[order]
                magnitudes = magnitudes[order]
                point_density = point_density[order]

            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=magnitudes,
                    y=semblances,
                    mode="markers",
                    name="Magnitude vs Semblance",
                    marker={
                        "color": point_density
                        if point_density is not None
                        else "black",
                        "colorscale": "Viridis",
                        "showscale": False,
                        "size": 10,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=uids,
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class MagnitudeRate(Component):
    name = "Magnitude Rate"
    description = """Magnitude of detected events over time. Size of markers corresponds
to magnitude value.
"""

    async def view(self) -> None:
        state = get_tab_state()
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
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot() -> None:
            catalog = await state.get_filtered_catalog()
            records = [
                (ev.time, ev.uid, ev.magnitude.average)
                for ev in catalog.events
                if ev.magnitude is not None
                and ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
            ]
            if not records:
                return
            times, uids, magnitudes = map(np.asarray, zip(*records, strict=True))
            magnitudes = np.asarray(magnitudes, dtype=float)
            # magnitudes, mask = magnitude_outlier_filer(magnitudes)
            # times = times[mask]
            # uids = uids[mask]
            if len(magnitudes) == 0:
                return

            # Keep a time-sorted copy for cumulative line computation.
            time_order = np.argsort(times)
            times_sorted = times[time_order]
            magnitudes_sorted = magnitudes[time_order]

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
                        "color": point_density
                        if point_density is not None
                        else "black",
                        "colorscale": "Viridis",
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
            moment_magnitudes = np.power(10, 1.5 * magnitudes_sorted + 9.1)
            cumulative_magnitudes = np.cumsum(moment_magnitudes)
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
            fig.update_layout(uirevision=True)

            plot.update()

        state.proxy_catalog.updated.subscribe(update_plot)
        background_tasks.create(update_plot())


class MagnitudeFrequencyBPositive(Component):
    name = "Magnitude Frequency b-Positive"
    description = """Frequency of positive magnitude differences between consecutive events, which can be used to estimate the b-value of the magnitude distribution."""

    async def _calculate_dmag_bpositive(
        self, times: np.ndarray, magnitudes: np.ndarray, d_mc: float
    ):
        idx = np.argsort(times)
        times_sorted = times[idx]
        magnitudes_sorted = magnitudes[idx]
        times_diff = times_sorted[:-1]
        mag_diff = magnitudes_sorted[1:] - magnitudes_sorted[:-1]
        idx_pos = mag_diff >= d_mc
        return times_diff[idx_pos], mag_diff[idx_pos] - d_mc

    async def view(self):
        state = get_tab_state()
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
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            events = [
                ev
                for ev in catalog.events
                if ev.magnitude is not None
                and ev.magnitude.average is not None
                and np.isfinite(ev.magnitude.average)
                and ev.magnitude.average >= 0
            ]
            magnitudes = np.asarray([ev.magnitude.average for ev in events])
            times = np.asarray([ev.time.timestamp() for ev in events])

            if len(magnitudes) == 0:
                return

            delta_mc = 0.5
            _, mag_diff = await self._calculate_dmag_bpositive(
                times,
                magnitudes,
                d_mc=delta_mc,
            )
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
                    marker={"color": "lightgray"},
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
                    line={"color": "red", "width": 1.5, "dash": "dash"},
                    hoverinfo="none",
                )
            )
            fig.update_layout(
                xaxis_title="ΔM",
                yaxis_title="Frequency",
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
            plot.update()

        state.proxy_catalog.updated.subscribe(update_plot)
        background_tasks.create(update_plot())


class StationMagnitudes(Component):
    name = "Station Magnitudes"
    description = """Magnitude values at individual stations for the selected event."""

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Magnitude",
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            sta_mags = [
                ev.magnitude.station_magnitudes
                for ev in catalog.events
                if ev.magnitude.station_magnitudes is not None
                and ev.magnitude.average is not None
            ]
            stds = [
                np.nanstd([sm.mag for sm in sms]) if len(sms) > 0 else np.nan
                for sms in sta_mags
            ]
            magnitudes = np.array(
                [
                    ev.magnitude.average
                    for ev in catalog.events
                    if ev.magnitude is not None and ev.magnitude.average is not None
                ]
            )
            fig.add_trace(
                go.Scattergl(
                    x=stds,
                    y=magnitudes,
                    mode="markers",
                    name="Station Magnitude Std Dev vs Event Magnitude",
                    marker={
                        "color": "black",
                        "size": 10,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    hoverinfo="none",
                    hovertemplate=None,
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class NPicksVsMagnitude(Component):
    name = "Number of Picks vs Magnitude"
    description = """Number of picks contributing to each event vs its magnitude."""

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Number of Picks",
            yaxis_title="Magnitude",
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            n_picks = np.asarray(
                [ev.n_picks for ev in catalog.events if ev.n_picks is not None],
                dtype=float,
            )
            magnitudes = np.asarray(
                [
                    ev.magnitude.average
                    for ev in catalog.events
                    if ev.magnitude is not None and ev.magnitude.average is not None
                ],
                dtype=float,
            )
            uids = np.asarray([str(ev.uid) for ev in catalog.events])
            if len(n_picks) == 0 or len(magnitudes) == 0:
                return
            # n_picks, mask = magnitude_outlier_filer(n_picks)
            # magnitudes = magnitudes[mask]
            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=n_picks,
                    y=magnitudes,
                    mode="markers",
                    name="Number of Picks vs Magnitude",
                    marker={
                        "color": "black",
                        "size": 10,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=uids,
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class SemblanceVsNPicks(Component):
    name = "Semblance vs Number of Picks"
    description = """Semblance value of detected events vs the number of picks contributing to them."""

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Number of Picks",
            yaxis_title="Semblance",
        )
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_events(plot)

        async def update_plot():
            catalog = await state.get_filtered_catalog()
            n_picks = np.asarray(
                [ev.n_picks for ev in catalog.events if ev.n_picks is not None],
                dtype=float,
            )
            semblances = np.asarray(catalog.semblances, dtype=float)
            finite_mask = np.isfinite(n_picks) & np.isfinite(semblances)
            n_picks = n_picks[finite_mask]
            semblances = semblances[finite_mask]
            uids = np.asarray([str(ev.uid) for ev in catalog.events])[finite_mask]
            if len(n_picks) == 0 or len(semblances) == 0:
                return
            # n_picks, mask = magnitude_outlier_filer(n_picks)
            # semblances = semblances[mask]
            plot.clear()
            fig.add_trace(
                go.Scattergl(
                    x=n_picks,
                    y=semblances,
                    mode="markers",
                    name="Semblance vs Number of Picks",
                    marker={
                        "color": "black",
                        "size": 10,
                        "line": {"width": 0},
                        "opacity": 0.3,
                    },
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=uids,
                )
            )
            plot.update()

        background_tasks.create(update_plot())


class StationMagnitudesResiduals(Component):
    name = "Station Magnitude Residuals"
    description = """Distance-corrected station magnitude residuals per station."""

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()

        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Station",
            yaxis_title="Magnitude Residual",
            showlegend=False,
        )

        plot = ui.plotly(fig).classes("w-full h-64")

        async def update_plot():
            catalog = await state.get_filtered_catalog()

            station_dict: dict[str, list[float]] = {}

            for ev in catalog.events:
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

            plot.clear()

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

            plot.update()

        background_tasks.create(update_plot())
