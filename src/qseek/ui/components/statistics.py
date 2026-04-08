from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from nicegui import background_tasks, ui

from qseek.ui.base import Component
from qseek.ui.state import get_tab_state
from qseek.ui.utils import on_click_plotly_event


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
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Semblance",
        )
        plot = (
            ui.plotly(fig)
            .classes("w-full h-64")
            .on("plotly_click", on_click_plotly_event)
        )

        async def update_plot():
            catalog = await state.run.get_catalog()
            semblances = catalog.semblances
            times = catalog.times

            plot.clear()
            fig.add_scatter(
                x=times,
                y=semblances,
                mode="markers",
                name="Event Semblance",
                hoverinfo="none",
                hovertemplate=None,
                customdata=catalog.uids,
                marker={
                    "color": "black",
                    "size": semblances / semblances.max() * 20,
                    "line": {"width": 0},
                    "opacity": 0.3,
                },
                # hovertemplate="Time: %{x}<br>Semblance: %{y:.3f}<extra></extra>",
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

        plot = (
            ui.plotly(fig)
            .classes("w-full h-64")
            .on("plotly_click", on_click_plotly_event)
        )

        async def update_plot():
            catalog = await state.run.get_catalog()
            distances = np.sqrt(
                catalog.north_shift**2 + catalog.east_shift**2 + catalog.depths**2
            )

            plot.clear()
            fig.add_scatter(
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
                    "size": catalog.semblances / catalog.semblances.max() * 20,
                    "line": {"width": 0},
                    "opacity": 0.3,
                },
            )
            plot.update()

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

        plot = (
            ui.plotly(fig)
            .classes("w-full h-64")
            .on("plotly_click", on_click_plotly_event)
        )

        async def update_plot():
            catalog = await state.run.get_catalog()

            if self.direction == "north-south":
                distances = catalog.north_shift
            else:
                distances = catalog.east_shift

            times = catalog.times
            times_num = np.array([(t - times[0]).total_seconds() for t in times]) / (
                3600 * 24
            )
            plot.clear()
            fig.add_scatter(
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
                    "size": catalog.semblances / catalog.semblances.max() * 20,
                    "line": {"width": 0},
                    "opacity": 0.3,
                },
            )
            plot.update()

        background_tasks.create(update_plot())
