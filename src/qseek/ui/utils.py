from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from nicegui import binding, ui
from nicegui.events import GenericEventArguments

T = TypeVar("T")

if TYPE_CHECKING:

    class BindableProperty(Generic[T]):
        """NiceGUI's BindableProperty, typed with the value of the property."""

        def __init__(self, on_change: Callable[..., Any] | None = None) -> None: ...
        def __set_name__(self, owner: type, name: str) -> None: ...
        def __get__(self, owner: object, _: object = None) -> T: ...
        def __set__(self, owner: object, value: T) -> None: ...

else:
    BindableProperty = binding.BindableProperty


def on_click_plotly_event(event: GenericEventArguments) -> None:
    points = event.args.get("points", [])
    if points:
        uid = points[0].get("customdata")
        if uid:
            ui.navigate.to(f"event/{uid}")


def attach_plotly_events(plot: ui.plotly) -> None:
    """Attach a plotly_click → navigate handler entirely in the browser.

    Reads `customdata` (event uid) from the clicked point and navigates to
    ``event/{uid}`` without a Python round-trip.  Safe to call after every
    ``plot.update()`` — the listener is replaced, not stacked.
    """
    ui.run_javascript(f"""
        (function attach(retries) {{
            var el = getElement({plot.id}).$el;
            if (!el) return;
            var gd = el.classList.contains('js-plotly-plot')
                ? el
                : el.querySelector('.js-plotly-plot');
            if (gd && typeof gd.on === 'function') {{
                gd.removeAllListeners('plotly_click');
                gd.on('plotly_click', function(data) {{
                    if (data.points && data.points.length > 0) {{
                        var uid = data.points[0].customdata;
                        if (uid) window.location.href = '/event/' + uid;
                    }}
                }});
            }} else if (retries > 0) {{
                setTimeout(function() {{ attach(retries - 1); }}, 100);
            }}
        }})(100);
    """)


# Animated radiating-wave event icon — matches logo_light.svg language:
# same color (#e4004b), ease-out expansion, pause after burst, stagger 1s.
# 40x40 SVG, center at (20,20), rings expand r=5→18, dur=6s (3s on + 3s pause).

_event_ring = (
    '<circle cx="20" cy="20" r="5" fill="none" stroke="#e4004b" stroke-width="2">'
    '<animate attributeName="r" values="5;18;18" keyTimes="0;0.5;1"'
    ' calcMode="spline" keySplines="0.2 0 0.8 1; 0 0 1 1"'
    ' dur="6s" repeatCount="indefinite" begin="{begin}"/>'
    '<animate attributeName="opacity" values="0.8;0;0" keyTimes="0;0.5;1"'
    ' calcMode="spline" keySplines="0.2 0 0.8 1; 0 0 1 1"'
    ' dur="6s" repeatCount="indefinite" begin="{begin}"/>'
    "</circle>"
)
EVENT_ANIMATED_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
    + _event_ring.format(begin="0s")
    + _event_ring.format(begin="1s")
    + _event_ring.format(begin="2s")
    + '<circle cx="20" cy="20" r="5" fill="#e4004b" stroke="black" stroke-width="1.5"/>'
    + "</svg>"
)


def card_header(title, description: str = "") -> None:
    ui.label(title).classes("text-h5")
    if description:
        ui.html(
            description,
            tag="div",
            sanitize=False,
        ).classes("text-body2 mb-2")


class StatCard(ui.card):
    def __init__(
        self,
        label: str,
        value: str = "",
        icon: str = "",
        subtitle: str = "",
        tooltip: str = "",
    ):
        super().__init__()
        with (
            self.classes("flex-1 min-w-40"),
            ui.column().classes("p-1 pb-0 gap-1 w-full"),
        ):
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon(icon).classes("text-lg text-grey-8")
                ui.label(label).classes(
                    "text-xs text-grey-8 uppercase tracking-wider font-semibold"
                )
                if tooltip:
                    ui.space()
                    with ui.icon("help_outline").classes("text-sm text-grey-5"):
                        ui.tooltip(tooltip).props("max-width=260px").classes("text-xs")
            self._value = ui.label(value).classes(
                "text-2xl font-bold text-grey-10 mt-1"
            )
            # Always render subtitle line to keep card height uniform
            self._subtitle = ui.label(subtitle or "\u00a0").classes(
                "text-xs text-grey-8 leading-tight"
            )

        self.bind_value = self._value.bind_text_from
        self.bind_subtitle = self._subtitle.bind_text_from


ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAP_ATTRIBUTION = (
    'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, HERE, Garmin,'
    " &copy; OpenStreetMap contributors, USGS, NGA, NASA, CGIAR"
    " and the GIS user community"
)


def add_basemap(m: ui.leaflet) -> None:
    """Replace the layers of a Leaflet map with a muted terrain basemap.

    The Esri Light Gray Canvas shows coastlines and roads, the Esri World Hillshade on
    top, multiplied into the canvas, adds the relief.
    """
    m.clear_layers()
    m.tile_layer(
        url_template=ESRI_TILES.format(service="Canvas/World_Light_Gray_Base"),
        options={
            "attribution": BASEMAP_ATTRIBUTION,
            "maxNativeZoom": 16,
            "maxZoom": 20,
        },
    )
    m.tile_layer(
        url_template=ESRI_TILES.format(service="Elevation/World_Hillshade"),
        options={
            "className": "qseek-hillshade",
            "maxNativeZoom": 16,
            "maxZoom": 20,
        },
    )
    ui.add_css(".qseek-hillshade { mix-blend-mode: multiply; }")
