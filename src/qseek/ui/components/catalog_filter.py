from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from nicegui import ui

from qseek.ui.state import get_tab_state

if TYPE_CHECKING:
    from qseek.ui.state import Filter


@dataclass(slots=True, kw_only=True)
class FilterDelegate:
    """Base delegate rendering a single filter control row."""

    icon: str
    label: str
    visible: Callable[[], bool] = lambda: True

    def header(self) -> None:
        with ui.row().classes("items-center gap-1.5"):
            ui.icon(self.icon, size="xs").classes("text-grey-5")
            ui.label(self.label).classes(
                "text-sm font-semibold text-grey-8 tracking-wide"
            )

    def build(self) -> None:
        raise NotImplementedError


@dataclass(slots=True, kw_only=True)
class RangeFilterDelegate(FilterDelegate):
    """Renders a min/max range slider bound to a `Filter`."""

    filter: Filter
    step: float
    formatter: Callable[[dict], str]
    color: str | None = None
    empty_message: str = "No data available"

    def build(self) -> None:
        with ui.column().classes("gap-2 w-full"):
            self.header()
            if not self.visible():
                ui.label(self.empty_message).classes("text-xs text-grey-4 italic")
                return

            range_slider = (
                ui.range(
                    min=self.filter.data_min,
                    max=self.filter.data_max,
                    step=self.step,
                    on_change=self.filter.set_user_defined,
                )
                .classes("w-full")
                .bind_value(self.filter, "range")
            )
            if self.color:
                range_slider.props(f"color={self.color}")
            ui.label().classes(
                "text-xs text-grey-6 font-mono text-right"
            ).bind_text_from(self.filter, "range", backward=self.formatter)


@dataclass(slots=True, kw_only=True)
class DateRangeFilterDelegate(FilterDelegate):
    """Renders a date-range picker behind a menu, bound to a `Filter`."""

    filter: Filter

    def build(self) -> None:
        with ui.column().classes("gap-2 w-full"):
            self.header()

            with ui.menu().props("no-parent-event") as date_menu:
                ui.date(on_change=self.filter.set_user_defined).props(
                    "range mask='YYYY-MM-DD' today-btn"
                ).bind_value(self.filter, "range")
                with ui.row().classes("justify-end px-2 pb-2"):
                    ui.button("OK", on_click=date_menu.close).props(
                        "flat dense size=sm color=primary"
                    )

            ui.button(icon="edit_calendar", on_click=date_menu.open).props(
                "outline color=teal icon-right"
            ).classes("w-full font-mono text-sm").bind_text_from(
                self.filter,
                "range",
                backward=lambda r: (
                    f"{r['from']}  →  {r['to']}"
                    if isinstance(r, dict) and "from" in r
                    else "Select range"
                ),
            )


def catalog_filter_dialog():
    state = get_tab_state()
    catalog = state.catalog_store

    if not catalog.has_catalog():
        ui.notify("Catalog is still loading, please wait...", type="warning")
        return

    delegates: list[FilterDelegate] = [
        RangeFilterDelegate(
            icon="stacked_line_chart",
            label="Semblance",
            filter=catalog.semblance_filter,
            step=0.01,
            color="primary",
            formatter=lambda r: f"{r['min']:.2f} - {r['max']:.2f}",
        ),
        RangeFilterDelegate(
            icon="bar_chart",
            label="Magnitude",
            filter=catalog.magnitude_filter,
            step=0.1,
            color="deep-orange",
            formatter=lambda r: f"{r['min']:.1f} - {r['max']:.1f}",
            visible=catalog.has_magnitudes,
            empty_message="No magnitudes in catalog",
        ),
        RangeFilterDelegate(
            icon="sensors",
            label="N Picks",
            filter=catalog.n_picks_filter,
            step=1.0,
            color="secondary",
            formatter=lambda r: f"{round(r['min'])} - {round(r['max'])}",
        ),
        RangeFilterDelegate(
            icon="sensors",
            label="RMS",
            filter=catalog.rms_filter,
            step=0.01,
            color="tertiary",
            formatter=lambda r: f"{r['min']:.2f} - {r['max']:.2f} s",
        ),
        RangeFilterDelegate(
            icon="height",
            label="Depth (m)",
            filter=catalog.depth_filter,
            step=10.0,
            formatter=lambda r: f"{r['min']:.0f} - {r['max']:.0f} m",
        ),
        DateRangeFilterDelegate(
            icon="calendar_month",
            label="Date Range",
            filter=catalog.time_filter,
        ),
    ]

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[760px] gap-0 !p-0 overflow-hidden rounded-xl shadow-2xl"),
    ):
        # ── Header ─────────────────────────────────────────────────────────
        with ui.row().classes("items-center w-full px-5 pt-5 pb-4 gap-3"):
            ui.icon("filter_alt", size="lg").classes("text-primary")

            with ui.column().classes("gap-0 flex-1"):
                ui.label("Filter Catalog").classes("text-base font-bold leading-snug")
                ui.label(
                    "Narrow events by quality, picks, magnitude, depth, and time"
                ).classes("text-xs text-grey-6 leading-tight")
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense size=sm color=grey-7"
            )

        ui.separator().classes("opacity-20")

        # ── Controls ───────────────────────────────────────────────────────
        with ui.column().classes("w-full px-6 py-5 gap-7"):
            for delegate in delegates:
                delegate.build()

        ui.separator().classes("opacity-20")

        # ── Footer ─────────────────────────────────────────────────────────
        with ui.row().classes("w-full px-6 py-4 items-center justify-between"):
            with ui.row().classes("items-center gap-1.5"):
                ui.icon("crisis_alert", size="xs").classes("text-grey-5")
                ui.label().classes("text-xs text-grey-6").bind_text_from(
                    catalog,
                    "events",
                    backward=lambda evs: f"{len(evs):,} events currently shown",
                )

            with ui.row().classes("gap-2"):

                def apply():
                    with state.loading_message("Applying filters..."):
                        dialog.close()
                        catalog.filter_events()
                        catalog.updated.emit()

                def do_reset():
                    catalog.reset_filters(reset_user_filters=True)
                    catalog.filter_events()
                    catalog.updated.emit()

                ui.button("Reset", on_click=do_reset).props(
                    "flat dense color=grey-7"
                ).classes("text-sm")
                ui.button("Apply", on_click=apply).props(
                    "unelevated color=primary"
                ).classes("text-sm px-4")

        dialog.open()
