from __future__ import annotations

from nicegui import Event, app, binding

from qseek.ui.explorer.base import RunSource


class TabState:
    run: RunSource
    run_name: str = binding.BindableProperty()
    _default_run: RunSource | None = None

    def __init__(self):
        if self._default_run is None:
            raise RuntimeError("No default run set")

        self.run = self._default_run
        self.run_name = self.run.name
        self.run_changed = Event()
        self.catalog_updated = Event()

    def set_run(self, run: RunSource):
        self.run = run
        self.run_name = run.name
        self.run_changed.emit()

    @classmethod
    def set_default_run(cls, run: RunSource):
        cls._default_run = run


def get_tab_state() -> TabState:
    if "state" not in app.storage.tab:
        app.storage.tab["state"] = TabState()
    return app.storage.tab["state"]
