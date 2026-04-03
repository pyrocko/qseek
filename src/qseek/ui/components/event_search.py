from __future__ import annotations

from nicegui import ui


class EventSearch:
    def render(self) -> None:
        results: list[dict] = []
        selected_idx = [0]

        with ui.element("div").classes("flex items-center"):
            with ui.menu().props("no-parent-event fit") as menu:

                @ui.refreshable
                def search_results() -> None:
                    with ui.element("q-list").props("separator").classes("min-w-96"):
                        if not results:
                            with (
                                ui.element("q-item"),
                                ui.element("q-item-section"),
                            ):
                                ui.label("No results").classes(
                                    "text-grey-6 text-sm italic"
                                )
                        for i, item in enumerate(results):
                            is_active = i == selected_idx[0]

                            def on_item_click(_, *, _item: dict = item) -> None:
                                ui.navigate(f"/event/{_item['id']}")
                                menu.close()

                            with (
                                ui.element("q-item")
                                .props(
                                    f"clickable v-ripple {'active active-class=bg-blue-grey-2' if is_active else ''}"
                                )
                                .on("click", on_item_click)
                            ):
                                with ui.element("q-item-section"):
                                    ui.label(item["time"]).classes("text-sm font-mono")
                                    ui.label(item["id"]).classes("text-xs text-grey-6")
                                with ui.element("q-item-section").props("side"):
                                    ui.label(f"M {item['magnitude']}").classes(
                                        "text-sm text-bold"
                                    )

                search_results()

            def on_search(value: str) -> None:
                if len(value) > 5:
                    # TODO: replace with actual search results
                    results.clear()
                    results.extend(
                        [
                            {
                                "time": "2024-03-15 08:42:11",
                                "id": "evt_00312",
                                "magnitude": "3.2",
                            },
                            {
                                "time": "2024-03-15 14:07:55",
                                "id": "evt_00318",
                                "magnitude": "1.8",
                            },
                        ]
                    )
                    selected_idx[0] = 0
                    search_results.refresh()
                    menu.open()
                else:
                    menu.close()

            def on_keydown(e) -> None:
                key = e.args.get("key", "") if isinstance(e.args, dict) else ""
                if key == "ArrowDown":
                    selected_idx[0] = (selected_idx[0] + 1) % max(len(results), 1)
                    search_results.refresh()
                elif key == "ArrowUp":
                    selected_idx[0] = (selected_idx[0] - 1) % max(len(results), 1)
                    search_results.refresh()
                elif key == "Enter" and results:
                    ui.navigate(f"/event/{results[selected_idx[0]]['id']}")
                    menu.close()
                elif key == "Escape":
                    menu.close()

            ui.input(
                placeholder="Search event by time or ID",
                on_change=lambda e: on_search(e.value),
            ).props("dark outlined dense color=white").classes("w-64").on(
                "keydown", on_keydown, throttle=0
            )
