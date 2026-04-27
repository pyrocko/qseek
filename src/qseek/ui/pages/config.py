import json

from nicegui import ui

from qseek.ui.state import get_tab_state


async def config_page() -> None:
    state = get_tab_state()
    search = await state.run.get_search()

    json_data = json.loads(search.model_dump_json())

    ui.json_editor({"content": {"json": json_data}}).classes("w-full")
