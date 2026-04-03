import json

import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

from qseek.ui.base import EventComponent


class ObservationsAzimuthsPlot(EventComponent):
    def __init__(self, event, phases: list[str]):
        super().__init__(event)
        self.phases = phases
        self.name = "Station Observations"
        self.description = (
            "Azimuthal plot of station observations for each phase. "
            "Colored diamonds = observed (color = delay), size = pick confidence. "
            "Hover for details. No modelled arrivals shown here since azimuths are "
            "not meaningful without a reference event.",
        )

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
            pick_confidence = [
                a.observed.detection_value if a and a.observed else 0 for a in arrivals
            ]
            has_obs = [bool(a and a.observed) for a in arrivals]
            max_delay = max(
                (abs(d) for d, obs in zip(delays, has_obs, strict=False) if obs),
                default=1.0,
            )
            phase_data[phase] = (delays, pick_confidence, has_obs, max_delay)

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
            delays, pick_confidence, has_obs, max_delay = phase_data[phase]
            cb_x = (col - 1) * (subplot_width + h_spacing) + subplot_width + 0.02
            fig.add_trace(
                go.Scatterpolar(
                    r=distances,
                    theta=azimuths,
                    mode="markers",
                    hovertext=[
                        f"<b>{label}</b><br>Distance: {dist:.1f} km<br>Azimuth: {az:.1f}°<br>"
                        + (
                            f"Confidence: {conf:.2f}<br>Delay: {delay:.3f} s"
                            if obs
                            else "No pick"
                        )
                        for label, dist, az, conf, delay, obs in zip(
                            labels,
                            distances,
                            azimuths,
                            pick_confidence,
                            delays,
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
                            "thickness": 15,
                            "x": cb_x,
                            "len": 0.9,
                            "y": 0.5,
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
            margin={"l": 60, "r": 80, "t": 30, "b": 20},
            template="plotly_white",
        )
        self.header()
        ui.plotly(fig).classes("w-full h-96")


class StationDistancesPlot(EventComponent):
    def __init__(self, event, phases: list[str]):
        super().__init__(event)
        self.phases = phases
        self.name = "Traveltime Residuals"
        self.description = (
            "Traveltime vs distance for all phase arrivals. "
            "Black circles = modelled, colored diamonds = observed (color = delay). "
            "Stems connect modelled to observed. Marker size = pick confidence."
        )

    async def view(self) -> None:
        ev = self.event
        phases = self.phases

        distances = [ev.surface_distance_to(r) / 1000.0 for r in ev.receivers]
        labels = [f"{r.network}.{r.station}.{r.location}" for r in ev.receivers]

        # Per-phase symbol pairs (modelled, observed)
        symbols = [
            ("circle", "diamond"),
            ("square", "star-diamond"),
            ("triangle-up", "cross"),
        ]

        # Collect per-phase data and find global max delay for shared colorscale
        phase_data = {}
        global_max_delay = 1.0
        for phase in phases:
            arrivals = [r.phase_arrivals.get(phase) for r in ev.receivers]
            modelled_tt = [
                (a.model.time - ev.time).total_seconds() if a else None
                for a in arrivals
            ]
            observed_tt = [
                (a.observed.time - ev.time).total_seconds()
                if a and a.observed
                else None
                for a in arrivals
            ]
            delays = [
                a.traveltime_delay.total_seconds() if a and a.traveltime_delay else 0.0
                for a in arrivals
            ]
            pick_confidence = [
                a.observed.detection_value if a and a.observed else 0.0
                for a in arrivals
            ]
            has_obs = [bool(a and a.observed) for a in arrivals]
            has_arrival = [a is not None for a in arrivals]
            phase_data[phase] = (
                modelled_tt,
                observed_tt,
                delays,
                pick_confidence,
                has_obs,
                has_arrival,
            )
            phase_max = max(
                (abs(d) for d, obs in zip(delays, has_obs, strict=False) if obs),
                default=0.0,
            )
            global_max_delay = max(global_max_delay, phase_max)

        fig = go.Figure()

        for idx, phase in enumerate(phases):
            modelled_tt, observed_tt, delays, pick_confidence, has_obs, has_arrival = (
                phase_data[phase]
            )
            sym_mod, sym_obs = symbols[idx % len(symbols)]
            phase_label = phase[-1]

            # Stems
            stem_x, stem_y = [], []
            for dist, mod_t, obs_t, obs in zip(
                distances, modelled_tt, observed_tt, has_obs, strict=False
            ):
                if obs:
                    stem_x.extend([dist, dist, None])
                    stem_y.extend([mod_t, obs_t, None])

            fig.add_trace(
                go.Scatter(
                    x=stem_x,
                    y=stem_y,
                    mode="lines",
                    line={"color": "grey", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=phase_label,
                )
            )

            # Modelled
            fig.add_trace(
                go.Scatter(
                    x=[d for d, a in zip(distances, has_arrival, strict=False) if a],
                    y=[t for t, a in zip(modelled_tt, has_arrival, strict=False) if a],
                    mode="markers",
                    name=f"{phase_label} modelled",
                    legendgroup=phase_label,
                    marker={
                        "symbol": sym_mod,
                        "size": 8,
                        "color": "black",
                        "opacity": [
                            1.0 if obs else 0.3
                            for obs, a in zip(has_obs, has_arrival, strict=False)
                            if a
                        ],
                    },
                    hovertext=[
                        f"<b>{label}</b><br>Distance: {dist:.1f} km<br>Modelled TT: {mod:.3f} s"
                        + ("" if obs else "<br>No pick")
                        for label, dist, mod, obs, a in zip(
                            labels,
                            distances,
                            modelled_tt,
                            has_obs,
                            has_arrival,
                            strict=False,
                        )
                        if a
                    ],
                    hovertemplate="%{hovertext}<extra>"
                    + phase_label
                    + " modelled</extra>",
                )
            )

            # Observed
            obs_zip = [
                (dist, tt, delay, conf, label)
                for dist, tt, delay, conf, label, obs in zip(
                    distances,
                    observed_tt,
                    delays,
                    pick_confidence,
                    labels,
                    has_obs,
                    strict=False,
                )
                if obs
            ]
            if obs_zip:
                o_dist, o_tt, o_delays, o_conf, o_labels = zip(*obs_zip, strict=False)
                is_last_phase = idx == len(phases) - 1
                fig.add_trace(
                    go.Scatter(
                        x=list(o_dist),
                        y=list(o_tt),
                        mode="markers",
                        name=f"{phase_label} observed",
                        legendgroup=phase_label,
                        marker={
                            "symbol": sym_obs,
                            "size": [max(c * 10, 5) for c in o_conf],
                            "color": list(o_delays),
                            "colorscale": "RdBu_r",
                            "cmin": -global_max_delay,
                            "cmax": global_max_delay,
                            "showscale": is_last_phase,
                            "colorbar": {"title": "Delay (s)", "thickness": 15},
                            "line": {"color": "black", "width": 1.2},
                        },
                        hovertext=[
                            f"<b>{label}</b><br>Distance: {dist:.1f} km<br>"
                            f"Observed TT: {tt:.3f} s<br>Confidence: {conf:.2f}<br>Delay: {delay:.3f} s"
                            for label, dist, tt, conf, delay in zip(
                                o_labels, o_dist, o_tt, o_conf, o_delays, strict=False
                            )
                        ],
                        hovertemplate="%{hovertext}<extra>"
                        + phase_label
                        + " observed</extra>",
                    )
                )

        fig.update_layout(
            xaxis={"title": "Distance (km)"},
            yaxis={"title": "Traveltime (s)"},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            margin={"l": 60, "r": 80, "t": 40, "b": 40},
            template="plotly_white",
        )
        self.header()
        ui.plotly(fig).classes("w-full h-96")


class EventMap(EventComponent):
    name = "Event Map"
    description = "Map showing event location and receiver stations."

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
            '<polygon points="8,0.5 14.5,11.75 1.5,11.75" fill="white" stroke="black" stroke-width="1.5"/>'
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
            group.addLayer(L.marker([{ev.effective_lat}, {ev.effective_lon}], {{icon: eventIcon}}));
            {json.dumps(stations)}.forEach(function(s) {{
                group.addLayer(L.marker([s.lat, s.lon], {{icon: stationIcon}})
                    .bindTooltip(s.label + '<br> Distance: ' + s.distance_km + ' km', {{permanent: false}}));
            }});
            group.addTo(map);
            map.fitBounds(group.getBounds(), {{padding: [30, 30]}});
        """)
