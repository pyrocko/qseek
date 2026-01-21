from __future__ import annotations

import inspect
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generator

from nicegui import ui
from nicegui.elements.card import Card

from qseek.ui.utils import snake_to_title

if TYPE_CHECKING:
    from pydantic import BaseModel


@contextmanager
def panel(classes: str = "", *, title: str = "") -> Generator[Card, Any, None]:
    with ui.card() as card:
        card.classes(f"p-4 {classes}")
        if title:
            ui.label(title).classes("text-lg text-slate-800")
        yield card


def item_pydantic(
    model: BaseModel,
    attribute: str,
    *,
    round_float: int = 2,
    prefix: str = "",
    suffix: str = "",
    title: str = "",
    description: str = "",
) -> None:
    try:
        field = model.model_fields[attribute]
        item_description = field.description
    except KeyError:
        value = getattr(model, attribute)
        try:
            item_description = inspect.getdoc(getattr(model.__class__, attribute))
        except AttributeError:
            item_description = ""

    title = title or snake_to_title(attribute)
    value = getattr(model, attribute)
    description = description or item_description

    if isinstance(value, float) and round_float >= 0:
        value = round(value, round_float)
    elif isinstance(value, datetime):
        value = value.strftime("%Y-%m-%d %H:%M") + f":{value.second:.2f} UTC"

    with ui.item().classes("col-4 col-md-4"):
        with ui.item_section():
            ui.item_label(title)
            if description:
                with ui.item_label(description).props("caption lines=1"):
                    ui.tooltip(description).props("delay=1000")
        with ui.item_section().props("side top"):
            label = f"{prefix}{value}{suffix}"
            ui.label(label).classes("text-bold text-md text-slate-700")
