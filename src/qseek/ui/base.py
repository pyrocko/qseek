from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

from qseek.ui.router import Router

if TYPE_CHECKING:
    from qseek.search import Search


ROUTER = Router()


def test():
    ui.label("test")


class Navigation:
    def __init__(self) -> None:
        with ui.left_drawer(bordered=True) as drawer:
            self.drawer = drawer
            with ui.list().classes("w-full") as list:
                self.list = list

    def add_page(self, name: str, icon: str, page) -> None:
        with self.list, ui.item(on_click=lambda: ROUTER.open(page)):
            with ui.item_section().props("avatar"):
                ui.icon(icon)
            with ui.item_section():
                ui.item_label(name)


class Header:
    def __init__(self) -> None:
        with ui.header(fixed=True, bordered=True).classes(
            "bg-white text-primary"
        ) as header:
            ui.label("QSeek").classes("text-base text-medium text-gray-900")
            ui.space()

            with ui.row().classes("items-center"):
                self.progress = ui.linear_progress(0.1, size="28px", show_value=True)
                self.progress.classes("w-64")
                self.progress.props("rounded stripe indeterminate")
                self.progress.set_visibility(False)

                self.live = ui.chip("Live", icon="ads_click", color="red")
                # self.live.set_visibility(False)

                self.run_name = ui.chip("Rundir", icon="folder")
                self.header = header

    def set_run_name(self, name: str):
        self.run_name.text = name

    def set_live(self, live: bool):
        self.live.set_visibility(live)


class FrontendPage:
    path: str
    name: str
    icon: str

    navigation: bool = True

    def __init__(self, search: Search) -> None:
        self.search = search

    def render(self, *args, **kwargs) -> None:
        self.ui(*args, **kwargs)

    def ui(self) -> None:
        for _ in range(100):
            ui.label("Hello World, overload me!")


class Site:
    _pages: list[FrontendPage]

    def __init__(self, search) -> None:
        self.header = Header()
        self.navigation = Navigation()
        self.search = search

        self._pages = []
        ROUTER.frame().classes("w-full p-4")

        for page in FrontendPage.__subclasses__():
            self.add_page(page(search=self.search))

    def add_page(self, page: FrontendPage) -> None:
        self._pages.append(page)
        if page.icon:
            self.navigation.add_page(page.name, page.icon, page.ui)
        ROUTER.add(page.path)(page.ui)
