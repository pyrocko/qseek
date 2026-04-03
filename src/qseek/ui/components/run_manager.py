from nicegui import ui

from qseek.ui.models import RunManager


def run_selection_dialog(manager: RunManager) -> None:
    active_run = manager.get_active_run()
    active_hash = active_run.hash if active_run else None

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
        with ui.row().classes("items-center justify-between w-full pb-2"):
            with ui.column().classes("gap-0"):
                ui.label("Switch Run").classes("text-lg text-bold")
                ui.label("Select a run to explore").classes("text-sm text-grey-6")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")

        columns = [
            {
                "name": "path",
                "label": "Run",
                "field": "path",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "created",
                "label": "Created",
                "field": "created",
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
                "path": run.path.name,
                "hash": run.hash,
                "created": run.created.strftime("%Y-%m-%d %H:%M"),
                "n_events": run.n_events,
            }
            for run in manager.runs.values()
        ]

        table = (
            ui.table(columns=columns, rows=rows).props("flat hover").classes("w-full")
        )
        table.props(
            f":row-class=\"row => row.hash === '{active_hash}' ? 'bg-blue-1 text-bold' : ''\""
        )

        table.on(
            "rowClick",
            lambda e: (manager.set_active_run(e.args[1]["hash"]), dialog.close()),
        )
        table.props("style='cursor: pointer'")

        dialog.open()
