import logging
import os
import subprocess
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


PLUGIN_MODULE = """
from typing import Literal

from qseek.plugins import Callback


class PluginCallback(Callback):
    callback: Literal["PluginCallback"] = "PluginCallback"
"""

CHECK_SEARCH = """
from pydantic import TypeAdapter

from qseek.search import Search

callbacks = TypeAdapter(Search.model_fields["callbacks"].annotation).validate_python(
    [{"callback": "PluginCallback"}]
)
print(type(callbacks[0]).__module__)
"""


def test_plugin_in_search_config(tmp_path):
    """A plugin importing from a package which builds a configuration type."""
    (tmp_path / "qseek_test_plugin.py").write_text(PLUGIN_MODULE)
    dist_info = tmp_path / "qseek_test_plugin-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: qseek-test-plugin\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[qseek.modules]\ntest = qseek_test_plugin\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-c", CHECK_SEARCH],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "qseek_test_plugin"
