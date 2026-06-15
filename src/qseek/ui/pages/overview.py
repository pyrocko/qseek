from __future__ import annotations

import numpy as np
from nicegui import background_tasks, ui
from nicegui.binding import bindable_dataclass

from qseek.ui.components.magnitudes import MagnitudeRate
from qseek.ui.components.map import OverviewMap
from qseek.ui.components.statistics import EventRate
from qseek.ui.state import CatalogStore, get_tab_state
from qseek.ui.utils import StatCard, card_header


@bindable_dataclass
class CatalogStats:
    n_events: int = 0
    n_days: float = 0.0
    event_rate: float = 0.0
    n_picks_median: float = 0.0
    n_picks_min: float = 0.0
    n_picks_max: float = 0.0
    rms_median_ms: float = 0.0
    rms_min_ms: float = 0.0
    rms_max_ms: float = 0.0
    magnitude_max: float = 0.0
    magnitude_min: float = 0.0

    def update_from_catalog(self, catalog: CatalogStore) -> None:
        self.n_events = catalog.n_events

        self.n_days = (
            (catalog.times[-1] - catalog.times[0]).total_seconds() / (3600 * 24)
            if catalog.n_events > 0
            else 0.0
        )
        self.event_rate = self.n_events / self.n_days if self.n_days > 0 else 1.0

        n_picks_values = [ev.n_picks for ev in catalog.events if ev.n_picks is not None]
        self.n_picks_median = np.nanmedian(n_picks_values) if n_picks_values else 0.0
        self.n_picks_min = np.nanmin(n_picks_values) if n_picks_values else 0.0
        self.n_picks_max = np.nanmax(n_picks_values) if n_picks_values else 0.0

        rms_values = np.array(
            [ev.event.rms for ev in catalog.events if ev.event.rms is not None],
            dtype=float,
        )
        if rms_values.size > 0:
            self.rms_median_ms = float(np.nanmedian(rms_values)) * 1e3
            self.rms_min_ms = float(np.nanmin(rms_values)) * 1e3
            self.rms_max_ms = float(np.nanmax(rms_values)) * 1e3
        else:
            self.rms_median_ms = 0.0
            self.rms_min_ms = 0.0
            self.rms_max_ms = 0.0

        self.magnitude_max = np.nanmax(catalog.magnitudes)
        self.magnitude_min = np.nanmin(catalog.magnitudes)


async def overview_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()
    search = await state.run.get_search()

    stats = CatalogStats()
    stats.update_from_catalog(catalog)
    catalog.updated.subscribe(lambda: stats.update_from_catalog(catalog))

    if catalog.n_events == 0:
        with ui.row().classes("items-center gap-2 text-grey-6 mt-2"):
            ui.icon("info").classes("text-grey-6")
            ui.label("No events found").classes("text-body1 font-medium")
        ui.label(
            "No events are present in this catalog yet. Run the detection pipeline to populate the catalog."
        ).classes("text-grey-6 text-body2")
        return

    with ui.row().classes("w-full items-stretch"):
        card_events = StatCard(
            "Total Events",
            value=stats.n_events,
            icon="crisis_alert",
            subtitle=f"over {stats.n_days:.0f} days",
            tooltip="Total number of detected events in the catalog.",
        )
        card_events.bind_value(stats, "n_events")
        card_events.bind_subtitle(
            stats,
            "n_days",
            backward=lambda v: f"over {v:.0f} days",
        )
        card_event_rate = StatCard(
            "Event rate",
            value="",
            subtitle="events/day",
            icon="timeline",
            tooltip="Average number of detected events per day.",
        )
        card_event_rate.bind_value(
            stats,
            "event_rate",
            backward=lambda _: f"{stats.event_rate:.2f}",
        )
        card_picks = StatCard(
            "Median Picks",
            f"{stats.n_picks_median:.0f}",
            icon="scatter_plot",
            subtitle=f"Min {stats.n_picks_min} / Max {stats.n_picks_max}",
            tooltip="Median number of picks per event.",
        )
        card_picks.bind_value(
            stats,
            "n_picks_median",
            backward=lambda v: f"{v:.0f}",
        )
        card_picks.bind_subtitle(
            stats,
            "n_picks_min",
            backward=lambda v: f"Min {v:.0f} / Max {stats.n_picks_max:.0f}",
        )

        if stats.rms_max_ms is not None:
            card_rms = StatCard(
                "Median RMS",
                f"{stats.rms_median_ms:.1f} ms",
                icon="manage_history",
                subtitle=f"Min {stats.rms_min_ms:.1f} / Max {stats.rms_max_ms:.1f} ms",
                tooltip="Median RMS of traveltime delays across all detected events.",
            )
            card_rms.bind_value(
                stats,
                "rms_median_ms",
                backward=lambda v: f"{v:.1f} ms",
            )
            card_rms.bind_subtitle(
                stats,
                "rms_min_ms",
                backward=lambda v: f"Min {v:.1f} / Max {stats.rms_max_ms:.1f} ms",
            )
        if catalog.has_magnitudes():
            card_magnitude = StatCard(
                "Max Magnitude",
                f"{stats.magnitude_max:.2f}",
                icon="bar_chart",
                subtitle=f"Min Magnitude: {stats.magnitude_min:.2f}",
                tooltip="Maximum magnitude among all detected events.",
            )
            card_magnitude.bind_value(
                stats,
                "magnitude_max",
                backward=lambda v: f"{v:.2f}",
            )
            card_magnitude.bind_subtitle(
                stats,
                "magnitude_min",
                backward=lambda v: f"Min Magnitude: {v:.2f}",
            )

    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12"):
            card_header(OverviewMap.title, OverviewMap.description)
            map_ = OverviewMap(catalog.lats.mean(), catalog.lons.mean())
            await map_.initialized()
            background_tasks.create(map_.add_stations(search.stations))
            background_tasks.create(map_.add_event_markers(catalog.events))

        with ui.card().classes("col-12"):
            card_header(MagnitudeRate.title, MagnitudeRate.description)
            mag_rate = MagnitudeRate()
            await mag_rate.plot_events(
                catalog.events,
                show_semblance=not catalog.has_magnitudes(),
                show_density=True,
            )

        with ui.card().classes("col-12"):
            card_header(EventRate.title, EventRate.description)
            event_rate = EventRate()
            await event_rate.add_events(catalog.events)
