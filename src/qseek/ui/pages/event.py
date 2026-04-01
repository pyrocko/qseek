from uuid import UUID

from nicegui import ui

from qseek.ui.base import Page


class EventPage(Page):
    async def render(self, event_id: str) -> None:
        run = self.run_manager.get_active_run()
        catalog = await run.get_catalog()
        event = catalog.get_event_by_uid(UUID(event_id))

        ui.label("Event Details").classes("text-h4 mb-4")
        ui.label(str(event))
