from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from qseek.ui.state import get_tab_state
from qseek.ui.utils import on_click_plotly_event

from qseek.ui.base import Component


class MagnitudeFrequency(Component):
    name = "Magnitude Frequency Distribution"
    description = """Frequency of detected events over magnitude bins."""

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
            catalog = await state.run.get_catalog()
            magnitudes = np.array(
                [
                    ev.magnitude.average
                    for ev in catalog.events
                    if ev.magnitude is not None and ev.magnitude.average is not None
                ]
            )

            mc_value, mc_params = await self._maximum_curvature(magnitudes)
            b_value, b_std_err = await self._b_positive_estimation(magnitudes)
            plot.clear()
            fig.add_bar(
                x=mc_params["bin_edges"][:-1],
                y=np.log10(mc_params["bin_counts"]),
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
                annotation_position="top right",
            )
            if not np.isnan(b_value):
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"b-value={b_value:.2f} ± {b_std_err:.2f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )
            plot.update()

        await update_plot()


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
        plot = ui.plotly(fig).classes("w-full h-64")

        async def update_plot():
            catalog = await state.run.get_catalog()
            magnitudes = np.asarray(
                [
                    ev.magnitude.average if ev.magnitude is not None else np.nan
                    for ev in catalog.events
                ],
                dtype=float,
            )
            semblances = np.asarray(catalog.semblances, dtype=float)
            mask = np.isfinite(magnitudes) & np.isfinite(semblances)
            magnitudes = magnitudes[mask]
            semblances = semblances[mask]

            plot.clear()
            fig.add_scatter(
                x=semblances,
                y=magnitudes,
                mode="markers",
                name="Magnitude vs Semblance",
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

        await update_plot()


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
        )
        plot = (
            ui.plotly(fig)
            .classes("w-full h-64")
            .on("plotly_click", on_click_plotly_event)
        )

        async def update_plot():
            catalog = await state.run.get_catalog()
            magnitudes = np.array(
                [
                    ev.magnitude.average
                    for ev in catalog.events
                    if ev.magnitude is not None and ev.magnitude.average is not None
                ]
            )
            times = np.array([ev.time for ev in catalog.events])

            plot.clear()
            min_mag = magnitudes.min() if len(magnitudes) > 0 else 0
            fig.add_scatter(
                x=times,
                y=magnitudes,
                mode="markers",
                name="Event Magnitude",
                customdata=[ev.uid for ev in catalog.events],
                marker={
                    "color": "black",
                    "size": (magnitudes - min_mag) / (magnitudes.max() - min_mag) * 20
                    if magnitudes.max() != min_mag
                    else 20,
                    "line": {"width": 0},
                    "opacity": 0.3,
                },
                hoverinfo="none",
                hovertemplate=None,
            )
            plot.update()

        await update_plot()


class StationMagnitudes(Component):
    name = "Station Magnitudes"
    description = """Magnitude values at individual stations for the selected event."""

    def __init__(self, event):
        self.event = event

    async def view(self) -> None:
        if self.event.magnitude is None or not self.event.magnitude.station_magnitudes:
            ui.label("No station magnitudes available for this event.").classes(
                "text-sm text-grey-6"
            )
            return

        fig = go.Figure()
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Station",
            yaxis_title="Magnitude",
        )
        plot = ui.plotly(fig).classes("w-full h-64")

        station_mags = self.event.magnitude.station_magnitudes
        stations = list(
            station_mags.station.network + "." + station_mags.station.station
        )
        magnitudes = list(station_mags.magnitude)

        fig.add_bar(
            x=stations,
            y=magnitudes,
            name="Station Magnitudes",
            marker_color="gray",
            hoverinfo="none",
            hovertemplate=None,
        )
        plot.update()
