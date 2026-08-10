from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_plugin(path: Path, factory: str = "load") -> Any:
    """Import a single-file plugin and instantiate it.

    The file is expected to define a top-level factory function returning
    the plugin instance.

    Args:
        path: Path to the plugin's `.py` file.
        factory: Name of the module-level factory function to call.

    Returns:
        The object returned by the plugin's factory function.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plugin file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, factory):
        raise AttributeError(f"plugin {path} does not define a `{factory}()` function")

    logger.info("loaded plugin from %s", path)
    return getattr(module, factory)()
