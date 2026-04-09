from __future__ import annotations

import math
from uuid import UUID

from nicegui import ui

from qseek.ui.base import Page
from qseek.ui.components.event import (
    EventMap,
    EventStationMagnitudes,
    ObservationsAzimuthsPlot,
    TravelTimeResidualPlot,
)
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


class EventPage(Page):
    async def render(self, event_id: str) -> None:
        run = get_tab_state().run
        catalog = await run.get_catalog()
        event = catalog.get_event_by_uid(UUID(event_id))
        ev = event.event

        depth_km = ev.depth / 1000.0

        # Header
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.button(icon="arrow_back", on_click=ui.navigate.back).props("flat round")
            ui.label(event.time.strftime("%Y-%m-%d  %H:%M:%S UTC")).classes(
                "text-h5 font-mono"
            )
            ui.space()
            ui.chip(str(event.uid), icon="fingerprint").props("outline").classes(
                "text-xs font-mono text-grey-6"
            )

        ui.separator().classes("mb-4")

        # Stat cards
        with ui.row().classes("w-full flex-wrap gap-2 mb-5"):
            mag = ev.magnitude
            if (
                mag is not None
                and mag.average is not None
                and not math.isnan(mag.average)
            ):
                mag_subtitle = (
                    f"± {mag.error:.2f}"
                    if mag.error is not None and not math.isnan(mag.error)
                    else ""
                )
                stat_card(
                    "Magnitude",
                    f"{mag.average:.2f}",
                    "speed",
                    subtitle=mag_subtitle,
                    tooltip=f"Network {mag.name} magnitude as computed as the median"
                    f" of station magnitudes ({len(mag.station_magnitudes)} stations). "
                    f"Error is the median absolute deviation.",
                )

            stat_card(
                "Semblance",
                f"{event.semblance:.3f}",
                icon="graphic_eq",
                tooltip="Normalized coherence of the stacked phase arrival beam "
                "in [0, 1]. Higher values indicate a more focused, confident "
                "detection.",
            )
            picks = ev.receivers.get_num_phase_picks()
            stat_card(
                "Picks",
                str(ev.n_picks),
                icon="network_ping",
                subtitle=f"{' / '.join(f'{ph[-1]} {n}' for ph, n in picks.items())}",
                tooltip="Number of phase arrivals associated with the event. Qseek "
                "associates picks based on their contribution to the semblance ",
            )
            stat_card(
                "Stations",
                str(ev.n_stations),
                icon="sensors",
                subtitle="Contributing stations",
                tooltip="Number of seismic stations online and contributing to the"
                " stacked phase arrival beam.",
            )
            # suf_lat = "E" if lon >= 0 else "W"
            # suf_lon = "N" if lat >= 0 else "S"
            # _stat_card(
            #     "Coordinates",
            #     f"{lat:.4f}°{suf_lat} {lon:.4f}°{suf_lon}",
            #     icon="explore",
            #     subtitle=f"± {(ev.uncertainty.horizontal / 1000):.2f} km"
            #     if ev.uncertainty
            #     else "",
            # )
            stat_card(
                "Depth",
                f"{depth_km:.2f} km",
                icon="vertical_align_bottom",
                subtitle=f"± {(ev.uncertainty.vertical / 1000):.2f} km"
                if ev.uncertainty
                else "",
                tooltip="Depth below the Earth's surface. Uncertainty (±) is "
                "derived from the 2% semblance threshold around the peak node.",
            )
            rms_phases = ev.receivers.get_rms()
            stat_card(
                "RMS",
                f"{ev.rms:.3f} s",
                icon="adjust",
                subtitle=f"{
                    ' / '.join(f'{p[-1]} {rms:.3f} s' for p, rms in rms_phases.items())
                }",
                tooltip="Root mean square of traveltime residuals between "
                "observed (picked) and modelled phase arrivals. Lower values indicate "
                "a better fit to the velocity model. Note that Qseek is using the "
                "full annotation information and is not working on picked arrival times",
            )
        with ui.row().classes("w-full flex-wrap gap-2 mb-5"):
            with ui.card().classes("w-full shadow-2 gap-2"):
                await EventMap(event.event).view()

            phases = list(ev.receivers.get_available_phases())

            with ui.card().classes("w-full flex-wrap gap-2"):
                await TravelTimeResidualPlot(event.event, phases).view()
            with ui.card().classes("w-full flex-wrap gap-2"):
                await ObservationsAzimuthsPlot(event.event, phases).view()

            for ev_mag in ev.magnitudes:
                with ui.card().classes("w-full flex-wrap gap-2"):
                    await EventStationMagnitudes(ev).view(ev_mag)
