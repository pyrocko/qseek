from __future__ import annotations

from nicegui import ui

from qseek.ui.state import get_tab_state


def catalog_filter_dialog():
    state = get_tab_state()
    catalog = state.catalog_store

    semblance_filter = catalog.semblance_filter
    magnitude_filter = catalog.magnitude_filter
    n_picks_filter = catalog.n_picks_filter
    rms_filter = catalog.rms_filter
    depth_filter = catalog.depth_filter
    time_filter = catalog.time_filter

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
            # ── Semblance ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("stacked_line_chart", size="xs").classes("text-grey-5")
                    ui.label("Semblance").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.range(
                    min=semblance_filter.data_min,
                    max=semblance_filter.data_max,
                    step=0.01,
                    on_change=lambda _: setattr(semblance_filter, "user_defined", True),
                ).classes("w-full").props("color=primary").bind_value(
                    semblance_filter, "range"
                )
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    semblance_filter,
                    "range",
                    backward=lambda r: f"{r['min']:.2f} - {r['max']:.2f}",
                )

            # ── Magnitude ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("bar_chart", size="xs").classes("text-grey-5")
                    ui.label("Magnitude").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                if catalog.has_magnitudes():
                    ui.range(
                        min=magnitude_filter.data_min,
                        max=magnitude_filter.data_max,
                        step=0.1,
                        on_change=lambda _: setattr(
                            magnitude_filter, "user_defined", True
                        ),
                    ).classes("w-full").props("color=deep-orange").bind_value(
                        magnitude_filter, "range"
                    )
                    ui.label().classes(
                        "text-xs text-grey-6 font-mono text-right"
                    ).bind_text_from(
                        magnitude_filter,
                        "range",
                        backward=lambda r: f"{r['min']:.1f} - {r['max']:.1f}",
                    )
                else:
                    ui.label("No magnitudes in catalog").classes(
                        "text-xs text-grey-4 italic"
                    )

            # ── N Picks ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("sensors", size="xs").classes("text-grey-5")
                    ui.label("N Picks").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.range(
                    min=n_picks_filter.data_min,
                    max=n_picks_filter.data_max,
                    step=1.0,
                    on_change=lambda _: setattr(n_picks_filter, "user_defined", True),
                ).classes("w-full").props("color=secondary").bind_value(
                    n_picks_filter, "range"
                )
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    n_picks_filter,
                    "range",
                    backward=lambda r: f"{round(r['min'])} - {round(r['max'])}",
                )

            # ── RMS ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("sensors", size="xs").classes("text-grey-5")
                    ui.label("RMS").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.range(
                    min=rms_filter.data_min,
                    max=rms_filter.data_max,
                    step=0.01,
                    on_change=lambda _: setattr(rms_filter, "user_defined", True),
                ).classes("w-full").props("color=tertiary").bind_value(
                    rms_filter, "range"
                )
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    rms_filter,
                    "range",
                    backward=lambda r: f"{r['min']:.2f} - {r['max']:.2f} s",
                )

            # ── Depth ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("height", size="xs").classes("text-grey-5")
                    ui.label("Depth (m)").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.range(
                    min=depth_filter.data_min,
                    max=depth_filter.data_max,
                    step=10.0,
                    on_change=lambda _: setattr(depth_filter, "user_defined", True),
                ).classes("w-full").bind_value(depth_filter, "range")
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    depth_filter,
                    "range",
                    backward=lambda r: f"{r['min']:.0f} - {r['max']:.0f} m",
                )

            # ── Date Range ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("calendar_month", size="xs").classes("text-grey-5")
                    ui.label("Date Range").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )

                with ui.menu().props("no-parent-event") as date_menu:
                    ui.date(
                        on_change=lambda _: setattr(time_filter, "user_defined", True),
                    ).props("range mask='YYYY-MM-DD' today-btn").bind_value(
                        time_filter, "range"
                    )
                    with ui.row().classes("justify-end px-2 pb-2"):
                        ui.button("OK", on_click=date_menu.close).props(
                            "flat dense size=sm color=primary"
                        )

                ui.button(icon="edit_calendar", on_click=date_menu.open).props(
                    "outline color=teal icon-right"
                ).classes("w-full font-mono text-sm").bind_text_from(
                    time_filter,
                    "range",
                    backward=lambda r: (
                        f"{r['from']}  →  {r['to']}"
                        if isinstance(r, dict) and "from" in r
                        else "Select range"
                    ),
                )

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
