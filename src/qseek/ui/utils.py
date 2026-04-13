from nicegui import ui
from nicegui.events import GenericEventArguments


def on_click_plotly_event(event: GenericEventArguments, route: str = "event") -> None:
    points = event.args.get("points", [])
    if points:
        target = points[0].get("customdata")
        if target:
            ui.navigate.to(f"/{route}/{target}")


def attach_plotly_navigate(
    plot: ui.plotly,
    route: str = "event",
    customdata_index: int | None = None,
) -> None:
    """Attach a plotly_click → navigate handler entirely in the browser.

    Reads `customdata` from the clicked point and navigates to
    ``{route}/{customdata}`` without a Python round-trip. If `customdata`
    contains an array, `customdata_index` selects which element to use.
    Safe to call after every ``plot.update()`` — the listener is replaced,
    not stacked.
    """
    ui.run_javascript(f"""
        (function attach(retries) {{
            var el = getElement({plot.id}).$el;
            if (!el) return;
            var gd = el.classList.contains('js-plotly-plot')
                ? el
                : el.querySelector('.js-plotly-plot');
            if (gd && typeof gd.on === 'function') {{
                gd.removeAllListeners('plotly_click');
                gd.on('plotly_click', function(data) {{
                    if (data.points && data.points.length > 0) {{
                        var target = data.points[0].customdata;
                        var idx = {customdata_index if customdata_index is not None else "null"};
                        if (Array.isArray(target) && idx !== null) {{
                            target = target[idx];
                        }}
                        if (target !== undefined && target !== null && target !== '') {{
                            window.location.href = '/{route}/' + encodeURIComponent(String(target));
                        }}
                    }}
                }});
            }} else if (retries > 0) {{
                setTimeout(function() {{ attach(retries - 1); }}, 100);
            }}
        }})(100);
    """)


def stat_card(
    label: str,
    value: str,
    icon: str,
    subtitle: str = "",
    tooltip: str = "",
) -> None:
    """Helper function to create a statistic card with consistent styling.

    Args:
        label: The label for the statistic (e.g. "Total Events").
        value: The value to display (e.g. "42").
        icon: The name of the Material Icon to display (e.g. "crisis_alert").
        subtitle: Optional subtitle to display below the value.
        tooltip: Optional tooltip text to display when hovering over the info icon.
    """
    with (
        ui.card().classes("flex-1 min-w-40 shadow-2"),
        ui.column().classes("p-1 pb-0 gap-1 w-full"),
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
