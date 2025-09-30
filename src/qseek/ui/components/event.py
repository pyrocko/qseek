from uuid import UUID

import plotly.express as px
from nicegui import ui

from qseek.magnitudes.base import EventMagnitude
from qseek.models.detection import EventDetection
from qseek.ui.base import FrontendPage
from qseek.ui.elements import item_pydantic, panel


class MagnitudeDetail:
    def __init__(self, magnitude: EventMagnitude) -> None:
        self.magnitude = magnitude

    def ui(self):
        magnitude = self.magnitude

        # magnitudes = np.array([sta.magnitude for sta in magnitude.station_magnitudes])
        # errors = np.array([sta.error for sta in magnitude.station_magnitudes])
        # distances = np.array([sta.distance_epi for sta in magnitude.station_magnitudes])

        ui.label(f"{magnitude.name}").classes("text-lg")
        with ui.row(wrap=True):
            item_pydantic(magnitude, "average", round_float=2)
            item_pydantic(magnitude, "error", round_float=2, prefix="±")
            item_pydantic(
                magnitude,
                "n_observations",
                title="Amplitude Measurements",
                round_float=2,
            )

        fig = px.scatter(
            x=[sta.distance_epi for sta in magnitude.station_magnitudes],
            y=[sta.magnitude for sta in magnitude.station_magnitudes],
            error_y=[sta.error for sta in magnitude.station_magnitudes],
            hover_name=[sta.station.pretty for sta in magnitude.station_magnitudes],
            labels={
                "x": "Distance from epicenter [km]",
                "y": "Station Magnitude",
            },
        )
        fig.add_hline(
            y=magnitude.average,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Magnitude {magnitude.average:.2f}",
        )

        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

        ui.plotly(fig).classes("w-full h-80")


class EventAttributes:
    def __init__(self, event: EventDetection):
        self.event = event

    def ui(
        self,
        uid: UUID,
    ) -> None:
        event = self.event

        ui.label("Event Details").classes("text-lg")
        with ui.row(wrap=True):
            item_pydantic(event, "time")
            item_pydantic(event, "semblance")
            item_pydantic(event, "lat", suffix="°", round_float=4)
            item_pydantic(event, "lon", suffix="°", round_float=4)
            item_pydantic(event, "depth", round_float=0, suffix=" m")
            if event.magnitudes:
                magnitude = event.magnitudes[0]
                item_pydantic(magnitude, "average", title=magnitude.name)

        map = ui.leaflet(
            center=(event.lat, event.lon),
            zoom=10,
        )
        map.generic_layer(
            name="circle",
            args=[
                (event.lat, event.lon),
                {"color": "red", "radius": event.semblance * 10},
            ],
        )


class EventDetails(FrontendPage):
    path = "/event/{uid}"
    name = "Event Detail"
    icon = ""

    def ui(self, uid: UUID):
        try:
            event = self.search.catalog.get_event(uid)
        except KeyError:
            ui.label("Event not found")
            return

        with ui.row():
            with panel("col-grow"):
                EventAttributes(event).ui(uid)
            for magnitude in event.magnitudes:
                with panel("col-5 col-md-5"):
                    MagnitudeDetail(magnitude).ui()
