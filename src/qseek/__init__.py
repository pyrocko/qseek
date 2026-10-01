"""qseek - data-driven earthquake detection and localization."""

from qseek.plugin_loader import load_plugins

# Plugins register their modules before the configuration models collect them
load_plugins()
