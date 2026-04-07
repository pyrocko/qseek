from __future__ import annotations

import asyncio

from nicegui import ui
from nicegui.events import KeyEventArguments

from qseek.ui.state import get_tab_state

_NAV_KEYS = {"ArrowDown", "ArrowUp", "Enter", "Escape"}
_PREVENT_DEFAULT_JS = (
    f"(e) => {{ if ({list(_NAV_KEYS)}.includes(e.key)) e.preventDefault(); }}"
)


class EventSearch:
    async def render(self) -> None:
        results: list[dict] = []
        selected_idx = 0

        with ui.element("div").classes("flex items-center w-72"):
            with ui.menu().props("no-parent-event no-focus fit") as menu:

                @ui.refreshable
                def search_results() -> None:
                    with ui.element("q-list").props("separator"):
                        if not results:
                            with ui.element("q-item"), ui.element("q-item-section"):
                                ui.label("No matching events found").classes(
                                    "text-grey-6 text-sm"
                                )
                                ui.label("Try a date (2024-03-15) or event ID").classes(
                                    "text-grey-8 text-xs"
                                )
                            return
                        for i, item in enumerate(results):
                            active_props = (
                                "active active-class=bg-blue-grey-1"
                                if i == selected_idx
                                else ""
                            )

                            def on_click(_, *, _item: dict = item) -> None:
                                ui.navigate.to(f"/event/{_item['uid']}")
                                menu.close()

                            with (
                                ui.element("q-item")
                                .props(f"clickable v-ripple {active_props}")
                                .on("click", on_click)
                            ):
                                with ui.element("q-item-section"):
                                    with ui.row().classes("items-center gap-2 no-wrap"):
                                        ui.label(
                                            f"{item['time'].strftime('%Y-%m-%d %H:%M:%S')} UTC"
                                        ).classes("text-sm text-bold font-mono")
                                    with ui.row().classes("items-center gap-2 no-wrap"):
                                        ui.icon("fingerprint").classes(
                                            "text-grey-6 text-s"
                                        )
                                        ui.label(f"{str(item['uid'])[:20]}…").classes(
                                            "text-xs text-grey-6 font-mono"
                                        )
                                with (
                                    ui.element("q-item-section").props("side"),
                                    ui.row().classes("items-baseline gap-1 no-wrap"),
                                ):
                                    if item["magnitude"] is not None:
                                        ui.label("M").classes("text-xs text-grey-6")
                                        ui.label(f"{item['magnitude']:.2f}").classes(
                                            "text-sm text-bold font-mono"
                                        )
                                    else:
                                        ui.label("S ").classes(
                                            "text-s text-grey-6 italic"
                                        )
                                        ui.label(f"{item['semblance']:.2f}").classes(
                                            "text-sm text-bold font-mono"
                                        )

                search_results()

            state = get_tab_state()

            async def on_search(value: str) -> None:
                nonlocal selected_idx
                if len(value) < 3:
                    results.clear()
                    search_results.refresh()
                    menu.close()
                    return

                catalog = await state.run.get_catalog()

                results.clear()
                for event in catalog.events:
                    if str(event.uid).startswith(value) or value in event.time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ):
                        results.append(
                            {
                                "uid": event.uid,
                                "time": event.time,
                                "magnitude": event.magnitude,
                                "semblance": event.semblance,
                            }
                        )
                    if len(results) >= 10:
                        break

                selected_idx = 0
                search_results.refresh()
                menu.open()

            ui.input(
                placeholder="Date or event ID  (Ctrl+K)",
                on_change=lambda e: asyncio.create_task(on_search(e.value)),
            ).props("dark outlined dense color=white").classes(
                "w-full search-event-input"
            ).on("keydown", js_handler=_PREVENT_DEFAULT_JS)

            # Ctrl+K must be handled client-side to beat browser interception
            ui.run_javascript("""
                document.addEventListener('keydown', function(e) {
                    if (e.ctrlKey && e.key === 'k') {
                        e.preventDefault();
                        var input = document.querySelector('.search-event-input input');
                        if (input) input.focus();
                    }
                });
            """)

            def on_key(e: KeyEventArguments) -> None:
                nonlocal selected_idx
                if not e.action.keydown or not menu.value:
                    return
                if e.key.name == "ArrowDown":
                    selected_idx = (selected_idx + 1) % max(len(results), 1)
                    search_results.refresh()
                elif e.key.name == "ArrowUp":
                    selected_idx = (selected_idx - 1) % max(len(results), 1)
                    search_results.refresh()
                elif e.key.name == "Enter" and results:
                    ui.navigate.to(f"/event/{results[selected_idx]['uid']}")
                    menu.close()
                elif e.key.name == "Escape":
                    menu.close()

            ui.keyboard(on_key=on_key, ignore=[])
