from __future__ import annotations

from nicegui import ui

from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state


def run_selection_dialog(manager: SourceManager) -> None:
    state = get_tab_state()
    active_hash = state.run.hash if state.run else None

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[560px] gap-0 !p-0 overflow-hidden rounded-xl shadow-2xl"),
    ):
        # ── Header ──────────────────────────────────────────────────────────
        with ui.row().classes("items-center w-full px-5 pt-5 pb-4 gap-3"):
            ui.icon("folder_open", size="sm").classes("text-primary")
            with ui.column().classes("gap-0 flex-1"):
                ui.label("Switch Run").classes("text-base font-bold leading-snug")
                ui.label("Select a run to explore").classes(
                    "text-xs text-grey-6 leading-tight"
                )
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense size=sm color=grey-7"
            )

        ui.separator().classes("opacity-20")

        # ── Run list ────────────────────────────────────────────────────────
        with (
            ui.scroll_area().classes("w-full").style("max-height: 420px"),
            ui.column().classes("w-full gap-0"),
        ):
            runs = sorted(
                manager.runs.values(),
                key=lambda r: r.last_update,
                reverse=True,
            )
            for run in runs:
                is_active = run.hash == active_hash
                card_classes = (
                    "w-full px-5 py-3.5 cursor-pointer gap-3 items-center "
                    "transition-colors duration-100 rounded-none border-0 "
                    + ("bg-blue-1" if is_active else "hover:bg-grey-1")
                )

                async def change_run(_, h=run.hash):
                    dialog.close()
                    await manager.set_active_run(h)

                with ui.row().classes(card_classes).on("click", change_run):
                    # Active indicator dot
                    ui.element("div").classes(
                        "w-2 h-2 rounded-full flex-shrink-0 "
                        + ("bg-primary" if is_active else "bg-transparent")
                    )

                    with ui.column().classes("gap-0.5 flex-1 min-w-0"):
                        ui.label(run.name).classes(
                            "text-sm font-medium truncate "
                            + (
                                "text-blue-9 font-semibold"
                                if is_active
                                else "text-grey-9"
                            )
                        )
                        ui.label(
                            f"Last updated {run.last_update.strftime('%Y-%m-%d %H:%M')}"
                        ).classes("text-xs text-grey-5 font-mono")

                    # Right side: source chip + event count (fixed widths to prevent jumping)
                    with ui.row().classes("items-center gap-3 flex-shrink-0"):
                        source_color = "positive" if run.source == "local" else "info"
                        ui.chip(run.source, icon="hub").props(
                            f"dense outline color={source_color}"
                        ).classes("text-xs w-16 justify-center")

                        ui.label(f"{run.n_events:,} events").classes(
                            "text-xs text-grey-6 font-mono text-right w-24"
                        )

                ui.separator().classes("opacity-10 mx-4")

        if not runs:
            with ui.column().classes("items-center gap-3 py-12 w-full"):
                ui.icon("folder_off", size="3em").classes("text-grey-3")
                ui.label("No runs loaded").classes("text-sm text-grey-5")

        dialog.open()
