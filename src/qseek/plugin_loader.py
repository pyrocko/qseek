"""Loader for module plugins.

Plugins are separate packages which provide additional modules, e.g. ray tracers
or station corrections, by subclassing qseek's module base classes. They are
registered as entry points in the `qseek.modules` group of their package:

```toml
[project.entry-points."qseek.modules"]
my_plugin = "my_plugin"
```

The plugins are imported when qseek is imported, before the configuration
models collect the available modules.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

PLUGIN_GROUP = "qseek.modules"

_LOADED_PLUGINS: dict[str, str] = {}
_PLUGINS_LOADED = False


def load_plugins() -> dict[str, str]:
    """Import all installed module plugins once.

    A plugin which fails to import is skipped with a warning.

    Returns:
        dict[str, str]: Names of the loaded plugins and their imported modules.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return _LOADED_PLUGINS
    _PLUGINS_LOADED = True

    for entry_point in entry_points(group=PLUGIN_GROUP):
        try:
            entry_point.load()
        except Exception as exc:
            logger.warning(
                "failed to load qseek plugin %s (%s): %s",
                entry_point.name,
                entry_point.value,
                exc,
            )
            continue
        _LOADED_PLUGINS[entry_point.name] = entry_point.module
        logger.debug(
            "loaded qseek plugin %s from %s", entry_point.name, entry_point.value
        )
    return _LOADED_PLUGINS


def get_plugin(module: type) -> str | None:
    """Get the plugin which provides a module class.

    Args:
        module (type): The module class.

    Returns:
        str | None: Name of the plugin, None for qseek's own modules.
    """
    package = module.__module__.split(".")[0]
    for name, plugin_module in _LOADED_PLUGINS.items():
        if plugin_module.split(".")[0] == package:
            return name
    return None
