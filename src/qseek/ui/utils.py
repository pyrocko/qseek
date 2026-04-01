from nicegui import ui
from nicegui.events import GenericEventArguments


def on_click_plotly_event(event: GenericEventArguments) -> None:
    points = event.args.get("points", [])
    if points:
        uid = points[0].get("customdata")
        if uid:
            ui.navigate.to(f"event/{uid}")


def stat_card(
    label: str,
    value: str,
    icon: str,
    subtitle: str = "",
    tooltip: str = "",
) -> None:
    with (
        ui.card().classes("flex-1 min-w-40 shadow-2"),
        ui.column().classes("p-2 gap-1 w-full"),
    ):
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon(icon).classes("text-lg text-grey-8")
            ui.label(label).classes(
                "text-xs text-grey-8 uppercase tracking-wider font-semibold"
            )
            if tooltip:
                ui.space()
                with ui.icon("help_outline").classes("text-sm text-grey-5"):
                    ui.tooltip(tooltip).props("max-width=260px").classes("text-xs")
        ui.label(value).classes("text-2xl font-bold text-grey-10 mt-1")
        # Always render subtitle line to keep card height uniform
        ui.label(subtitle or "\u00a0").classes("text-xs text-grey-8 leading-tight")
