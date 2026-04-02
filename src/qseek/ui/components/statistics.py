import plotly.graph_objects as go
from nicegui import ui
import numpy as np

from qseek.ui.base import Component
from qseek.ui.utils import on_click_plotly_event


class SemblanceRate(Component):
    name = "Semblance Rate"
    description = """
Semblance of detected events over time. Size of markers corresponds
to semblance value.
"""

    async def view(self) -> None:
        catalog = await self.run.get_catalog()
        semblances = catalog.semblances
        times = catalog.times

        fig = go.Figure()
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        fig.add_scatter(
            x=times,
            y=semblances,
            mode="markers",
            name="Event Semblance",
            customdata=catalog.uids,
            marker={
                "color": "black",
                "size": semblances / semblances.max() * 20,
                "line": {"width": 0},
                "opacity": 0.3,
            },
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Semblance",
        )

        ui.plotly(fig).classes("w-full h-64").on("plotly_click", on_click_plotly_event)


class MagnitudeRate(Component):
    name = "Magnitude Rate"
    description = """Magnitude of detected events over time. Size of markers corresponds
to magnitude value.
"""

    async def view(self) -> None:
        catalog = await self.run.get_catalog()
        magnitudes = catalog.magnitudes
        times = catalog.times

        fig = go.Figure()
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        fig.add_scatter(
            x=times,
            y=magnitudes,
            mode="markers",
            name="Event Magnitude",
            marker={
                "color": "black",
                "size": magnitudes / magnitudes.max() * 20,
                "line": {"width": 0},
                "opacity": 0.3,
            },
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Magnitude",
        )
        ui.plotly(fig).classes("w-full h-64")


class StationCoverage(Component):
    name = "Station Coverage"
    description = """Number of stations contributing to each detected event."""

    async def view(self) -> None: ...


class MigrationPlot(Component):
    name = "Migration Plot"
    description = """Plot distance to center over time to visualize event migration. Size of markers corresponds to magnitude and color corresponds to depth."""

    async def view(self) -> None:
        catalog = await self.run.get_catalog()
        depths = catalog.depths
        distances = catalog.distances_to_center
        times = catalog.times

        fig = go.Figure()
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        fig.add_scatter(
            x=times,
            y=distances,
            mode="markers",
            name="Migration Plot",
            marker={
                "color": depths,
                "colorscale": "Magma",
                "size": catalog.magnitudes / catalog.magnitudes.max() * 20,
                "line": {"width": 0},
                "opacity": 0.3,
            },
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Distance to Center (km)",
        )

        ui.plotly(fig).classes("w-full h-64")


class DepthSection(Component):
    @property
    def name(self) -> str:
        axis = "North-South" if self.direction == "north-south" else "East-West"
        return f"Depth Section ({axis})"

    description = """Depth of detected events along profile line through the center. Color corresponds to time and size corresponds to magnitude."""

    def __init__(self, run, direction: str = "north-south") -> None:
        super().__init__(run)
        self.direction = direction

    async def view(self) -> None:
        catalog = await self.run.get_catalog()
        depths = catalog.depths
        times = catalog.times
        times_num = np.array([(t - times[0]).total_seconds() for t in times]) / (
            3600 * 24
        )

        if self.direction == "north-south":
            distances = catalog.north_shift
        else:
            distances = catalog.east_shift

        fig = go.Figure()
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        fig.add_scatter(
            x=distances,
            y=-depths,
            mode="markers",
            name=self.name,
            marker={
                "color": times_num,
                "colorscale": "Magma",
                "size": catalog.magnitudes / catalog.magnitudes.max() * 20,
                "line": {"width": 0},
                "opacity": 0.3,
            },
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Distance to Center (km)",
            yaxis_title="Depth (m)",
        )

        ui.plotly(fig).classes("w-full h-64")
