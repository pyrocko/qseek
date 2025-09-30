import plotly.graph_objects as go
from nicegui import ui

from qseek.ui.base import Component


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
        ui.plotly(fig).classes("w-full h-64")


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
