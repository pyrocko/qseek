import logging
import sys
import types
from importlib.metadata import EntryPoint

import pytest

from qseek import plugin_loader
from qseek.tracers.constant_velocity import ConstantVelocityTracer


@pytest.fixture
def plugins(monkeypatch):
    """Fake installed plugins: a working one and one failing on import."""
    module = types.ModuleType("qseek_test_plugin")
    module.PluginModule = type("PluginModule", (), {"__module__": module.__name__})
    monkeypatch.setitem(sys.modules, module.__name__, module)

    entry_points = [
        EntryPoint("test", "qseek_test_plugin", plugin_loader.PLUGIN_GROUP),
        EntryPoint("broken", "qseek_missing_plugin", plugin_loader.PLUGIN_GROUP),
    ]
    monkeypatch.setattr(
        plugin_loader,
        "entry_points",
        lambda group: [ep for ep in entry_points if ep.group == group],
    )
    monkeypatch.setattr(plugin_loader, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(plugin_loader, "_LOADED_PLUGINS", {})
    return module


def test_load_plugins(plugins, caplog):
    with caplog.at_level(logging.WARNING):
        loaded = plugin_loader.load_plugins()

    assert loaded == {"test": "qseek_test_plugin"}
    assert "failed to load qseek plugin broken" in caplog.text
    # Plugins are loaded once
    assert plugin_loader.load_plugins() is loaded


def test_get_plugin(plugins):
    plugin_loader.load_plugins()
    assert plugin_loader.get_plugin(plugins.PluginModule) == "test"
    assert plugin_loader.get_plugin(ConstantVelocityTracer) is None
