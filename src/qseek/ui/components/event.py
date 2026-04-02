import plotly.graph_objects as go
from nicegui import ui

from qseek.ui.base import EventComponent


class StationAzimuthsPlot(EventComponent):
    name = "Station Azimuths"
    description = "Azimuths from the event to each receiver station."

    async def view(self) -> None:
        ev = self.event

        distances = [
            ev.surface_distance_to(receiver) / 1000.0 for receiver in ev.receivers
        ]
        azimuths = [ev.azimuth_to(receiver) for receiver in ev.receivers]
        labels = [f"{r.network}.{r.station}.{r.location}" for r in ev.receivers]
        picks = [r.n_picks() for r in ev.receivers]
        fig = go.Figure()
        scatter = go.Scatterpolar(
            r=distances,
            theta=azimuths,
            mode="markers",
            text=labels,
            customdata=picks,
            textposition="top center",
            textfont={"size": 10},
            marker={
                "symbol": "diamond",
                "size": [max(p * 10, 5) for p in picks],
                "color": "orange",
                "line": {"color": "black", "width": 1},
            },
            hovertemplate="<b>%{text}</b><br>Distance: %{r:.1f} km<br>"
            "Azimuth: %{theta:.1f}°<br>"
            "N Picks: %{customdata}<extra></extra>",
        )
        fig.add_trace(scatter)
        fig.update_layout(
            polar={
                "angularaxis": {
                    "rotation": 90,
                    "direction": "clockwise",
                    "tickmode": "array",
                    "tickvals": [0, 45, 90, 135, 180, 225, 270, 315],
                    "ticktext": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                },
                "radialaxis": {
                    "ticksuffix": " km",
                    "showticklabels": True,
                    "tickangle": 45,
                },
            },
            margin={"l": 60, "r": 60, "t": 20, "b": 20},
            showlegend=False,
            template="plotly_white",
        )
        ui.plotly(fig).classes("w-full h-96")


class EventMap(EventComponent):
    name = "Event Map"
    description = "Map showing event location and receiver stations."

    async def view(self) -> None:
        ev = self.event

        m = ui.leaflet(
            center=(ev.lat, ev.lon),
            zoom=9,
        ).classes("w-full h-96 rounded-lg shadow")
        m.marker(latlng=(ev.lat, ev.lon))
        for receiver in ev.receivers:
            r_lat, r_lon = receiver.lat, receiver.lon
            m.marker(latlng=(r_lat, r_lon), options={"opacity": 0.4})
