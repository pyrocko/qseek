from nicegui import ui

from qseek.ui.models import RunManager


def run_selection_dialog(manager: RunManager) -> None:
    with ui.dialog() as dialog, ui.card().tight():
        table = ui.table(
            columns=[
                {
                    "name": "path",
                    "label": "Run Name",
                    "field": "path",
                    "sortable": True,
                    "align": "left",
                },
                {
                    "name": "action",
                    "label": "",
                    "field": "action",
                    "sortable": False,
                    "align": "right",
                },
            ],
            rows=[
                {"path": run.path.name, "hash": run.hash}
                for run in manager.runs.values()
            ],
        ).props("striped hover")

        active_run = manager.get_active_run()
        active_hash = active_run.hash if active_run else None
        if active_hash:
            table.props(
                f":row-class=\"row => row.hash === '{active_hash}' ? 'bg-blue-2' : ''\""
            )

        def select_run(hash: str) -> None:
            manager.set_active_run(hash)
            dialog.close()

        with table.add_slot("body-cell-action"), table.cell("action"):
            ui.button(icon="arrow_forward").props("flat round").on(
                "click",
                js_handler="() => emit(props.row.hash)",
                handler=lambda e: select_run(e.args),
            )
        dialog.open()
