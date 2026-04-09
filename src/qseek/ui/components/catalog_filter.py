from nicegui import ui


def catalog_filter_dialog():
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label("Catalog Filter").classes("text-h2")
        with ui.row().classes("gap-4"):
            ...
        with ui.row().classes("gap-2 justify-end mt-4"):
            ui.button("Apply", on_click=dialog.close).props("unelevated")

        dialog.open()
