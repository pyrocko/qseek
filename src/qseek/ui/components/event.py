import json

import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

from qseek.magnitudes.base import EventStationMagnitude
from qseek.ui.base import EventComponent


class ObservationsAzimuthsPlot(EventComponent):
    def __init__(self, event, phases: list[str]):
        super().__init__(event)
        self.phases = phases
        self.name = "Station Observations"
        self.description = """
Azimuthal plot of station observations for each phase. Colored diamonds = observed
(color is delay), size is pick confidence. No modelled arrivals shown here since
azimuths are not meaningful without a reference event.
"""

    async def view(self) -> None:
        ev = self.event
        phases = self.phases
        n = len(phases)

        distances = [ev.surface_distance_to(r) / 1000.0 for r in ev.receivers]
        azimuths = [ev.azimuth_to(r) for r in ev.receivers]
        labels = [f"{r.network}.{r.station}.{r.location}" for r in ev.receivers]

        phase_data = {}
        for phase in phases:
            arrivals = [r.phase_arrivals.get(phase) for r in ev.receivers]
            delays = [
                a.traveltime_delay.total_seconds() if a and a.traveltime_delay else 0
                for a in arrivals
            ]
            modelled_tt = [
                (a.model.time - ev.time).total_seconds() if a else None
                for a in arrivals
            ]
            pick_confidence = [
                a.observed.detection_value if a and a.observed else 0 for a in arrivals
            ]
            has_obs = [bool(a and a.observed) for a in arrivals]
            max_delay = max(
                (abs(d) for d, obs in zip(delays, has_obs, strict=False) if obs),
                default=1.0,
            )
            phase_data[phase] = (
                delays,
                modelled_tt,
                pick_confidence,
                has_obs,
                max_delay,
            )

        polar_axis_config = {
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": [45, 90, 135, 180, 225, 270, 315],
                "ticktext": ["NE", "E", "SE", "S", "SW", "W", "NW"],
            },
            "radialaxis": {
                "ticksuffix": " km",
                "showticklabels": True,
                "tickangle": 45,
            },
        }

        h_spacing = 0.2
        subplot_width = (1.0 - h_spacing * (n - 1)) / n

        fig = make_subplots(
            rows=1,
            cols=n,
            specs=[[{"type": "polar"}] * n],
            subplot_titles=[f"{phase[-1]} Phase" for phase in phases],
            horizontal_spacing=h_spacing,
        )

        for col, phase in enumerate(phases, start=1):
            delays, modelled_tt, pick_confidence, has_obs, max_delay = phase_data[phase]
            # Center each colorbar under its subplot
            cb_x = (col - 1) * (subplot_width + h_spacing) + subplot_width / 2
            fig.add_trace(
                go.Scatterpolar(
                    r=distances,
                    theta=azimuths,
                    mode="markers",
                    hovertext=[
                        f"<b>{label}</b> ({dist:.1f} km; {az:.1f}°)<br>"
                        + f"Confidence: {conf:.2f}<br>"
                        + (
                            f"Delay: {delay:+.3f} s ({delay / tt * 100:+.1f}%)<br>"
                            if obs and tt
                            else "No pick<br>"
                        )
                        + f"Total TT: {tt:.2f} s<br>"
                        for label, dist, az, conf, delay, tt, obs in zip(
                            labels,
                            distances,
                            azimuths,
                            pick_confidence,
                            delays,
                            modelled_tt,
                            has_obs,
                            strict=False,
                        )
                    ],
                    marker={
                        "symbol": "diamond",
                        "size": [max(v * 10, 5) for v in pick_confidence],
                        "color": delays,
                        "colorscale": "RdBu_r",
                        "cmin": -max_delay,
                        "cmax": max_delay,
                        "showscale": True,
                        "colorbar": {
                            "title": f"{phase[-1]} Delay (s)",
                            "orientation": "h",
                            "thickness": 12,
                            "x": cb_x,
                            "xanchor": "center",
                            "y": -0.12,
                            "yanchor": "top",
                            "len": subplot_width,
                            "tickformat": "+.2f",
                        },
                        "opacity": [1.0 if obs else 0.5 for obs in has_obs],
                        "line": {"color": "black", "width": 1.2},
                    },
                    hovertemplate="%{hovertext}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            polar_key = "polar" if col == 1 else f"polar{col}"
            fig.update_layout(**{polar_key: polar_axis_config})

        fig.update_layout(
            margin={"l": 60, "r": 40, "t": 30, "b": 60},
            template="plotly_white",
        )
        self.header()
        ui.plotly(fig).classes("w-full h-96")


class TravelTimeResidualPlot(EventComponent):
    def __init__(self, event, phases: list[str]):
        super().__init__(event)
        self.phases = phases
        self.name = "Travel Time Residuals"
        self.description = """
Traveltime residuals (<i>t<sub>observed</sub> - t<sub>modelled</sub></i>) per
 phase vs. epicentral distance. Marker size scales with pick confidence. Color encodes
 the residual: red = late arrival, blue = early arrival. Open markers at zero indicate
 stations with no pick for that phase.
"""

    async def view(self) -> None:
        ev = self.event
        phases = self.phases

        distances = [ev.surface_distance_to(r) / 1000.0 for r in ev.receivers]
        labels = [f"{r.network}.{r.station}.{r.location}" for r in ev.receivers]

        _symbols = ["circle", "diamond", "square", "cross"]

        phase_data = {}
        global_max_delay = 0.0
        for phase in phases:
            arrivals = [r.phase_arrivals.get(phase) for r in ev.receivers]
            delays = [
                a.traveltime_delay.total_seconds() if a and a.traveltime_delay else 0.0
                for a in arrivals
            ]
            modelled_tt = [
                (a.model.time - ev.time).total_seconds() if a else None
                for a in arrivals
            ]
            pick_confidence = [
                a.observed.detection_value if a and a.observed else 0.0
                for a in arrivals
            ]
            has_obs = [bool(a and a.observed) for a in arrivals]
            phase_data[phase] = (delays, modelled_tt, pick_confidence, has_obs)
            phase_max = max(
                (abs(d) for d, obs in zip(delays, has_obs, strict=False) if obs),
                default=0.0,
            )
            global_max_delay = max(global_max_delay, phase_max)
        global_max_delay = global_max_delay or 1.0  # fallback if no observations

        fig = go.Figure()

        for idx, phase in enumerate(phases):
            delays, modelled_tt, pick_confidence, has_obs = phase_data[phase]
            symbol = _symbols[idx % len(_symbols)]
            phase_label = phase[-1]

            # Observed — colored by residual, opacity by confidence
            obs_data = [
                (dist, delay, tt, conf, label)
                for dist, delay, tt, conf, label, obs in zip(
                    distances,
                    delays,
                    modelled_tt,
                    pick_confidence,
                    labels,
                    has_obs,
                    strict=True,
                )
                if obs
            ]
            if obs_data:
                o_dist, o_delays, o_tt, o_conf, o_labels = zip(*obs_data, strict=True)
                fig.add_trace(
                    go.Scatter(
                        x=list(o_dist),
                        y=list(o_delays),
                        mode="markers",
                        name=phase_label,
                        legendgroup=phase_label,
                        marker={
                            "symbol": symbol,
                            "size": [max(c * 14, 5) for c in o_conf],
                            "color": list(o_delays),
                            "colorscale": "RdBu_r",
                            "cmin": -global_max_delay,
                            "cmax": global_max_delay,
                            "showscale": False,
                            "line": {"color": "black", "width": 1.2},
                        },
                        hovertext=[
                            f"<b>{phase_label} - {label}</b> ({dist:.1f} km)<br>"
                            f"Confidence: {conf:.2f}<br>"
                            f"Residual: {delay:+.3f} s ({delay / tt * 100:+.1f}%)<br>"
                            f"Total TT: {tt:.2f} s"
                            for label, dist, delay, tt, conf in zip(
                                o_labels, o_dist, o_delays, o_tt, o_conf, strict=True
                            )
                        ],
                        hovertemplate="%{hovertext}<extra></extra>",
                    )
                )

            # No picks — open markers pinned to zero
            nopick_data = [
                (dist, label)
                for dist, label, obs in zip(distances, labels, has_obs, strict=True)
                if not obs
            ]
            if nopick_data:
                n_dist, n_labels = zip(*nopick_data, strict=True)
                fig.add_trace(
                    go.Scatter(
                        x=list(n_dist),
                        y=[0.0] * len(n_dist),
                        mode="markers",
                        name=f"{phase_label} (no pick)",
                        legendgroup=phase_label,
                        showlegend=False,
                        marker={
                            "symbol": symbol + "-open",
                            "size": 6,
                            "color": "grey",
                            "opacity": 0.7,
                        },
                        hovertext=[
                            f"<b>{label} Phase</b><br>"
                            f"Distance: {dist:.1f} km<br>"
                            f"Phase: {phase_label}<br>"
                            f"No pick"
                            for label, dist in zip(n_labels, n_dist, strict=True)
                        ],
                        hovertemplate="%{hovertext}<extra></extra>",
                    )
                )

        fig.add_hline(y=0, line={"color": "grey", "width": 1, "dash": "dot"})

        fig.update_layout(
            xaxis={"title": "Distance (km)"},
            yaxis={"title": "Residual (s)", "zeroline": False},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            margin={"l": 60, "r": 40, "t": 40, "b": 40},
            template="plotly_white",
        )
        self.header()
        ui.plotly(fig).classes("w-full h-80")


class EventMap(EventComponent):
    name = "Event Map"
    description = """
Map showing event location and online stations contributing to the location.
"""

    async def view(self) -> None:
        ev = self.event

        self.header()
        m = ui.leaflet(center=(ev.effective_lat, ev.effective_lon), zoom=9).classes(
            "w-full h-96 rounded-lg shadow"
        )

        stations = [
            {
                "lat": r.effective_lat,
                "lon": r.effective_lon,
                "label": f"{r.network}.{r.station}.{r.location}",
                "distance_km": round(ev.surface_distance_to(r) / 1000.0, 1),
            }
            for r in ev.receivers
        ]

        # Animated radiating-wave event icon — matches logo_light.svg language:
        # same color (#e4004b), ease-out expansion, pause after burst, stagger 1s.
        # 40x40 SVG, center at (20,20), rings expand r=5→18, dur=6s (3s on + 3s pause).
        _ring = (
            '<circle cx="20" cy="20" r="5" fill="none" stroke="#e4004b" stroke-width="2">'
            '<animate attributeName="r"       values="5;18;18" keyTimes="0;0.5;1"'
            ' calcMode="spline" keySplines="0.2 0 0.8 1; 0 0 1 1"'
            ' dur="6s" repeatCount="indefinite" begin="{begin}"/>'
            '<animate attributeName="opacity" values="0.8;0;0" keyTimes="0;0.5;1"'
            ' calcMode="spline" keySplines="0.2 0 0.8 1; 0 0 1 1"'
            ' dur="6s" repeatCount="indefinite" begin="{begin}"/>'
            "</circle>"
        )
        event_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
            + _ring.format(begin="0s")
            + _ring.format(begin="1s")
            + _ring.format(begin="2s")
            + '<circle cx="20" cy="20" r="5" fill="#e4004b" stroke="black" stroke-width="1.5"/>'
            + "</svg>"
        )

        # Equilateral triangle (side=13, centered in 16x16): apex(8,0.5), base(1.5,11.75)-(14.5,11.75)
        station_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
            '<polygon points="8,0.5 14.5,11.75 1.5,11.75" fill="#5C8FA3" stroke="black" stroke-width="1.5"/>'
            "</svg>"
        )

        await m.initialized()

        ui.run_javascript(f"""
            var map = getElement({m.id}).map;

            var eventIcon = L.divIcon({{
                html: '{event_svg}',
                iconSize: [50, 50],
                iconAnchor: [25, 25],
                className: ''
            }});
            var stationIcon = L.divIcon({{
                html: '{station_svg}',
                iconSize: [16, 16],
                iconAnchor: [8, 8],
                className: ''
            }});
            var group = L.featureGroup();
            group.addLayer(L.marker([{ev.effective_lat}, {ev.effective_lon}], {{icon: eventIcon}})
                .bindTooltip('<b>Event</b><br>{ev.time.strftime("%Y-%m-%d %H:%M:%S")} UTC<br>Lat: {ev.effective_lat:.4f}°<br>Lon: {ev.effective_lon:.4f}°<br>Depth: {ev.effective_depth / 1000.0:.1f} km', {{permanent: false}}));
            {json.dumps(stations)}.forEach(function(s) {{
                group.addLayer(L.marker([s.lat, s.lon], {{icon: stationIcon}})
                    .bindTooltip(s.label + '<br> Distance: ' + s.distance_km + ' km', {{permanent: false}}));
            }});
            group.addTo(map);
            map.fitBounds(group.getBounds(), {{padding: [30, 30]}});
        """)


class EventStationMagnitudes(EventComponent):
    name = "Station Magnitudes"

    description = """
Station magnitudes for each phase. Only shown if station magnitudes are available for the event.
"""

    async def view(self, magnitudes: EventStationMagnitude) -> None:
        self.header()

        if not magnitudes:
            ui.label("No station magnitudes available for this event.").classes(
                "text-gray-6"
            )
            return

        return

        station_mags = []
        for sta_mag in magnitudes.station_magnitudes:
            station_mags.append(
                {
                    "distance_epi": sta_mag.distance_epi,
                    "distnace_hypo": sta_mag.distance_hypo,
                    "magnitude": sta_mag.magnitude,
                }
            )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[m["station"] for m in station_mags],
                    y=[m["magnitude"] for m in station_mags],
                    text=[m["phase"] for m in station_mags],
                    marker_color="#5C8FA3",
                )
            ]
        )
        fig.update_layout(
            title="Station Magnitudes",
            xaxis_title="Station",
            yaxis_title="Magnitude",
            margin={"l": 60, "r": 40, "t": 40, "b": 80},
            template="plotly_white",
        )
        self.header()
        ui.plotly(fig).classes("w-full h-80")
