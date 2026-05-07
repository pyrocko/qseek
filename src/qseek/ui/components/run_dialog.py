from __future__ import annotations

from nicegui import ui

from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state


def run_selection_dialog(manager: SourceManager) -> None:
    state = get_tab_state()
    active_hash = state.run.hash if state.run else None

    columns = [
        {
            "name": "name",
            "label": "Run",
            "field": "name",
            "align": "left",
            "sortable": True,
        },
        {
            "name": "source",
            "label": "Source",
            "field": "source",
            "align": "center",
            "sortable": True,
        },
        {
            "name": "last_update_ts",
            "label": "Last Modified",
            "field": "last_update_ts",
            "sortable": True,
            "align": "right",
        },
        {
            "name": "n_events",
            "label": "Events",
            "field": "n_events",
            "sortable": True,
            "align": "right",
        },
        {
            "name": "action",
            "label": "",
            "field": "action",
            "align": "center",
            "sortable": False,
        },
    ]

    rows = sorted(
        [
            {
                "hash": run.hash,
                "name": run.name,
                "source": run.source,
                "last_update_ts": int(run.last_update.timestamp()),
                "last_update_str": run.last_update.strftime("%b %d, %Y  %H:%M"),
                "n_events": run.n_events,
                "is_active": run.hash == active_hash,
                "tags": getattr(run, "tags", []),
            }
            for run in manager.runs.values()
        ],
        key=lambda r: r["last_update_ts"],
        reverse=True,
    )

    with (
        ui.dialog() as dialog,
        ui.card().classes(
            "!max-w-none gap-0 !p-0 overflow-hidden rounded-xl shadow-2xl"
        ),
    ):
        # ── Header ──────────────────────────────────────────────────────────
        with ui.row().classes("items-center w-full px-5 pt-5 pb-4 gap-3"):
            ui.icon("folder_open", size="lg").classes("text-primary")
            with ui.column().classes("gap-0 flex-1"):
                ui.label("Switch Run").classes("text-base font-bold leading-snug")
                ui.label("Select a run to explore").classes(
                    "text-xs text-grey-6 leading-tight"
                )
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense size=sm color=grey-7"
            )

        ui.separator().classes("opacity-20")

        # ── Empty state ──────────────────────────────────────────────────────
        if not rows:
            with ui.column().classes("items-center gap-3 py-12 w-full"):
                ui.icon("folder_off", size="3em").classes("text-grey-3")
                ui.label("No runs loaded").classes("text-sm text-grey-5")
        else:
            # ── Table ────────────────────────────────────────────────────────
            # virtual-scroll requires a fixed height class (h-*), not max-height
            table = (
                ui.table(
                    columns=columns,
                    rows=rows,
                    row_key="hash",
                    pagination={
                        "rowsPerPage": 0,
                        "sortBy": "last_update_ts",
                        "descending": True,
                    },
                )
                .classes("w-full h-[480px]")
                .props("flat dense virtual-scroll hide-bottom")
            )

            table.add_slot(
                "body-cell-name",
                r"""
                <q-td :props="props">
                    <div class="flex items-center gap-2.5">
                        <div :class="[
                            'w-1.5 h-1.5 rounded-full flex-shrink-0',
                            props.row.is_active ? 'bg-primary' : 'bg-transparent'
                        ]" />
                        <span :class="[
                            'text-sm',
                            props.row.is_active
                                ? 'text-primary font-semibold'
                                : 'text-grey-9 font-medium'
                        ]">{{ props.row.name }}</span>
                    </div>
                </q-td>
                """,
            )

            table.add_slot(
                "body-cell-source",
                r"""
                <q-td :props="props" class="text-center">
                    <div v-if="props.row.source === 'qseek-http'" class="flex items-center justify-center gap-1.5">
                        <span class="relative flex h-2 w-2">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                        </span>
                        <span class="text-xs text-red-500 font-medium">live</span>
                    </div>
                    <q-chip v-else
                        outline size="sm"
                        :color="{local:'positive',ssh:'info'}[props.row.source] || 'grey'"
                        :icon="{local:'computer',ssh:'terminal'}[props.row.source] || 'circle'"
                    >{{ props.row.source }}</q-chip>
                </q-td>
                """,
            )

            table.add_slot(
                "body-cell-last_update_ts",
                r"""
                <q-td :props="props" class="text-right text-xs text-grey-6">
                    {{ props.row.last_update_str }}
                </q-td>
                """,
            )

            table.add_slot(
                "body-cell-n_events",
                r"""
                <q-td :props="props" class="text-right text-xs font-medium">
                    {{ props.row.n_events.toLocaleString() }}
                </q-td>
                """,
            )

            async def on_selected(e) -> None:
                dialog.close()
                await manager.set_active_run(e.args)

            with table.add_slot("body-cell-action"), table.cell("action"):
                ui.button("Select").props("dense size=sm color=primary").on(
                    "click",
                    js_handler="() => emit(props.row.hash)",
                    handler=on_selected,
                )

        dialog.open()
