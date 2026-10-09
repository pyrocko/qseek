from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

from qseek.plugins.base import load_plugin
from qseek.plugins.callback import Callback
from qseek.plugins.telegram import TelegramAlert  # noqa: F401

logger = logging.getLogger(__name__)

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type CallbackType = Callback
else:
    type CallbackType = Annotated[
        Union[Callback.get_subclasses()],
        Field(discriminator="callback"),
    ]


def load_callback_plugin(path: Path) -> Callback:
    """Load a `Callback` from a single-file plugin.

    The file is expected to define a top-level `load()` function
    returning a `Callback` instance.

    Args:
        path: Path to the plugin's `.py` file.

    Returns:
        Callback: The loaded callback instance.
    """
    callback = load_plugin(path)
    if not isinstance(callback, Callback):
        raise TypeError(f"plugin {path} did not return a Callback instance")
    logger.info("loaded callback plugin from %s", path)
    return callback
