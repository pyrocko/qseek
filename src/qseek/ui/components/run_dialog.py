from nicegui import ui

from qseek.ui.manager import SourceManager
from qseek.ui.state import get_tab_state


def run_selection_dialog(manager: SourceManager) -> None:
    state = get_tab_state()
    active_run = state.run
    active_hash = active_run.hash if active_run else None

    with (
        ui.dialog().props("w-max-96") as dialog,
        ui.card().classes("w-auto min-w-96"),
    ):
        with (
            ui.row().classes("items-center justify-between w-full pb-2"),
            ui.column().classes("gap-0"),
        ):
            ui.label("Switch Run").classes("text-lg text-bold")
            ui.label("Select a run to explore").classes("text-sm text-grey-6")

        columns = [
            {
                "name": "path",
                "label": "Run",
                "field": "path",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "source",
                "label": "Source",
                "field": "source",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "last_modified",
                "label": "Last Modified",
                "field": "last_modified",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "n_events",
                "label": "Events",
                "field": "n_events",
                "sortable": True,
                "align": "right",
            },
        ]

        rows = [
            {
                "path": run.name,
                "hash": run.hash,
                "source": run.source,
                "last_modified": run.last_update.strftime("%Y-%m-%d %H:%M"),
                "n_events": run.n_events,
            }
            for run in manager.runs.values()
        ]

        table = (
            ui.table(columns=columns, rows=rows)
            .props("flat hover sort-by=last_modified :sort-descending=true")
            .classes("w-full")
        )
        table.add_slot(
            "body-cell-source",
            """
            <q-td :props="props">
                <q-chip
                    dense outline
                    :color="props.row.source === 'local' ? 'green' : 'light-blue'"
                    text-color="white"
                    class="text-xs"
                >
                    {{ props.row.source }}
                </q-chip>
            </q-td>
            """,
        )
        table.props(
            f":row-class=\"row => row.hash === '{active_hash}' ? 'bg-blue-1 text-bold' : ''\""
        )

        async def change_run(e):
            selected_hash = e.args[1]["hash"]
            dialog.close()
            await manager.set_active_run(selected_hash)

        table.on("rowClick", change_run)
        table.props("style='cursor: pointer'")

        table.add_slot(
            "empty",
            """
            <div class="q-pa-md flex flex-col items-center gap-2">
                <q-icon name="folder_open" size="3em" color="grey-5" />
                <div class="text-grey-5">No runs loaded</div>
            </div>
            """,
        )

        dialog.open()
