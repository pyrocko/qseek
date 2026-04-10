from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui
from scipy.stats import gaussian_kde

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import attach_plotly_navigate


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
    description = """Frequency of detected events over magnitude bins."""

    def _b_value_fit_line(
        self,
        bin_edges: np.ndarray,
        bin_counts: np.ndarray,
        b_value: float,
        mc_value: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = (bin_counts > 0) & (centers >= mc_value)
        if np.count_nonzero(mask) < 2:
            return None

        x = centers[mask]
        y = np.log10(bin_counts[mask])
        slope = -b_value
        intercept = float(np.mean(y - slope * x))

        fit_x = np.linspace(x.min(), x.max(), 100)
        fit_y = slope * fit_x + intercept
        return fit_x, fit_y, slope, intercept

    async def _maximum_curvature(self, magnitudes: np.ndarray) -> Tuple[float, Dict]:
        # Create magnitude bins
        bin_width = 0.1
        min_mag = np.floor(magnitudes.min() * 10) / 10
        max_mag = np.ceil(magnitudes.max() * 10) / 10
        bins = np.arange(min_mag - bin_width / 2, max_mag + bin_width / 2, bin_width)

        # Calculate histogram
        counts, bin_edges = np.histogram(magnitudes, bins=bins)

        # Find bin with maximum count
        max_idx = np.argmax(counts)
        mc_value = bin_edges[max_idx]

        params = {
            "bin_counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "max_count": int(counts[max_idx]),
        }

        return mc_value, params

    async def _b_positive_estimation(self, magnitudes, delta_m=0.1):
        mags = np.asarray(magnitudes)
        mags, _ = magnitude_outlier_filer(mags)
        if len(mags) < 2:
            return np.nan, np.nan
        dm = np.diff(mags)
        dm_pos = dm[dm > 0]
        if len(dm_pos) < 5:
            return np.nan, np.nan
        mean_dm = np.mean(dm_pos)
        if mean_dm <= delta_m / 2:
            return np.nan, np.nan
        b_hat = np.log10(np.e) / (mean_dm - delta_m / 2)
        std_err = b_hat / np.sqrt(len(dm_pos))
        return float(b_hat), float(std_err)

    async def view(self) -> None:
        state = get_tab_state()
        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Magnitude",
            yaxis_title="log(Frequency)",
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
            magnitudes, _ = magnitude_outlier_filer(magnitudes)
            mc_value, mc_params = await self._maximum_curvature(magnitudes)
            b_value, b_std_err = await self._b_positive_estimation(magnitudes)
            plot.clear()
            counts = np.asarray(mc_params["bin_counts"], dtype=float)
            bin_edges = np.asarray(mc_params["bin_edges"], dtype=float)
            fig.add_bar(
                x=mc_params["bin_edges"][:-1],
                y=np.log10(counts),
                name="Magnitude Distribution",
                marker_color="gray",
                hoverinfo="none",
                hovertemplate=None,
            )
            fig.add_vline(
                x=mc_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"MC={mc_value:.2f}",
                annotation_position="top left",
            )

            if not np.isnan(b_value):
                fit = self._b_value_fit_line(bin_edges, counts, b_value, mc_value)
                if fit is not None:
                    fit_x, fit_y, _, _ = fit
                    fig.add_scatter(
                        x=fit_x,
                        y=fit_y,
                        mode="lines",
                        name="b-value fit",
                        line={"color": "blue", "width": 2},
                        hoverinfo="none",
                        hovertemplate=None,
                    )
                fig.add_annotation(
                    x=1.38,
                    y=0.7,
                    xref="paper",
                    yref="paper",
                    text=f"b-value={b_value:.2f} ± {b_std_err:.2f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )
            plot.update()

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
            xaxis_title="Semblance",
            yaxis_title="Magnitude",
        )
        # fig.update_yaxes(scaleanchor="x", scaleratio=1)
        plot = ui.plotly(fig).classes("w-full h-64")
        attach_plotly_navigate(plot)

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
            magnitudes, mask = magnitude_outlier_filer(magnitudes)
            semblances = semblances[mask]
            uids = uids[mask]

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
            fig.add_scatter(
                x=magnitudes,
                y=semblances,
                mode="markers",
                name="Magnitude vs Semblance",
                marker={
                    "color": point_density if point_density is not None else "black",
                    "colorscale": "Viridis",
                    "showscale": point_density is not None,
                    "colorbar": {"title": "Density"},
                    "size": 10,
                    "line": {"width": 0},
                    "opacity": 0.1,
                },
                hoverinfo="none",
                hovertemplate=None,
                customdata=uids,
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
            margin={"l": 0, "r": 80, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Magnitude",
            legend={
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.75)",
            },
            yaxis2={
                "title": "Cumulative Magnitude",
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
            magnitudes, mask = magnitude_outlier_filer(magnitudes)
            times = times[mask]
            uids = uids[mask]
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

            plot.clear()
            min_mag = scatter_magnitudes.min() if len(scatter_magnitudes) > 0 else 0
            fig.add_scatter(
                x=scatter_times,
                y=scatter_magnitudes,
                mode="markers",
                name="Event Magnitude",
                customdata=scatter_uids,
                marker={
                    "color": point_density if point_density is not None else "black",
                    "colorscale": "Viridis",
                    "showscale": point_density is not None,
                    "colorbar": {
                        "title": "Density",
                        "x": 1.1,
                        "xanchor": "left",
                    },
                    "size": (scatter_magnitudes - min_mag)
                    / (scatter_magnitudes.max() - min_mag)
                    * 15
                    if scatter_magnitudes.max() != min_mag
                    else 10,
                    "line": {"width": 0},
                    "opacity": 0.1,
                },
                hoverinfo="none",
                hovertemplate=None,
            )
            cumulative_magnitudes = np.cumsum(magnitudes_sorted)
            fig.add_scatter(
                x=times_sorted,
                y=cumulative_magnitudes,
                mode="lines",
                name="Cumulative Magnitude",
                line={"color": "red", "dash": "solid", "width": 3},
                hoverinfo="none",
                hovertemplate=None,
                yaxis="y2",
            )

            plot.update()

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
            fig.add_scatter(
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
            plot.update()

        background_tasks.create(update_plot())
