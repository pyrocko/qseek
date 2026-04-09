from __future__ import annotations

from nicegui import ui

from qseek.ui.state import get_tab_state


def catalog_filter_dialog():
    state = get_tab_state()
    catalog = state.filtered_catalog

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[500px] gap-0 !p-0 overflow-hidden rounded-xl shadow-2xl"),
    ):
        # ── Header ─────────────────────────────────────────────────────────
        with ui.row().classes("items-center w-full px-5 pt-5 pb-4 gap-3"):
            ui.element("div").classes(
                "flex items-center justify-center w-8 h-8 rounded-lg"
            ).style("background: rgba(25,118,210,0.10)")
            ui.icon("filter_alt", size="sm").classes("text-primary absolute")

            with ui.column().classes("gap-0 flex-1"):
                ui.label("Filter Catalog").classes("text-base font-bold leading-snug")
                ui.label("Narrow events by quality, magnitude, and time").classes(
                    "text-xs text-grey-6 leading-tight"
                )
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense size=sm color=grey-7"
            )

        ui.separator().classes("opacity-20")

        # ── Controls ───────────────────────────────────────────────────────
        with ui.column().classes("w-full px-6 py-5 gap-7"):
            # ── Semblance ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("stacked_line_chart", size="xs").classes("text-grey-5")
                    ui.label("Semblance").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.range(
                    min=0.0,
                    max=2.0,
                    step=0.01,
                ).classes("w-full").props(
                    "color=primary :label-always='true'"
                ).bind_value(catalog, "semblance_range")

            # ── Magnitude ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("bar_chart", size="xs").classes("text-grey-5")
                    ui.label("Magnitude").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                if catalog.has_magnitudes():
                    ui.range(
                        min=-1,
                        max=9,
                        step=0.1,
                    ).classes("w-full").props(
                        "color=deep-orange :label-always='true'"
                    ).bind_value(catalog, "magnitude_range")
                else:
                    ui.label("No magnitudes in catalog").classes(
                        "text-xs text-grey-4 italic"
                    )

            # ── Date Range ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("calendar_month", size="xs").classes("text-grey-5")
                    ui.label("Date Range").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )

                with ui.menu().props("no-parent-event") as date_menu:
                    ui.date().props("range mask='YYYY-MM-DD' today-btn").bind_value(
                        catalog, "date_range"
                    )
                    with ui.row().classes("justify-end px-2 pb-2"):
                        ui.button("OK", on_click=date_menu.close).props(
                            "flat dense size=sm color=primary"
                        )

                ui.button(icon="edit_calendar", on_click=date_menu.open).props(
                    "outline color=teal icon-right"
                ).classes("w-full font-mono text-sm").bind_text_from(
                    catalog,
                    "date_range",
                    backward=lambda r: f"{r['from']}  \u2192  {r['to']}"
                    if isinstance(r, dict) and "from" in r
                    else "Select range",
                )

        ui.separator().classes("opacity-20")

        # ── Footer ─────────────────────────────────────────────────────────
        with ui.row().classes("w-full px-6 py-4 items-center justify-between"):
            with ui.row().classes("items-center gap-1.5"):
                ui.icon("crisis_alert", size="xs").classes("text-grey-5")
                ui.label(f"{catalog.n_events:,} events currently shown").classes(
                    "text-xs text-grey-6"
                )

            with ui.row().classes("gap-2"):

                def apply():
                    catalog._refresh_event_data()
                    dialog.close()

                ui.button("Reset", on_click=catalog.reset_filters).props(
                    "flat dense color=grey-7"
                ).classes("text-sm")
                ui.button("Apply", on_click=apply).props(
                    "unelevated color=primary"
                ).classes("text-sm px-4")

        dialog.open()
