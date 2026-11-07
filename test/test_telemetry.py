from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import posthog
import pytest

from qseek import telemetry
from qseek.tracers.cake import CakeTracer
from qseek.tracers.constant_velocity import ConstantVelocityTracer


class FakePosthog:
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events: list[tuple[str, dict[str, Any]]] = []

    def capture(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def flush(self, timeout_seconds: float | None = None) -> None:
        pass


@pytest.fixture
def telemetry_env(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "CONFIG_FILE", tmp_path / "telemetry.json")
    monkeypatch.setattr(telemetry, "POSTHOG_API_KEY", "phc_test")
    monkeypatch.setattr(telemetry, "_client", None)
    monkeypatch.setattr(telemetry, "_install_id", None)
    monkeypatch.setattr(posthog, "Posthog", FakePosthog)

    def clear_environment() -> None:
        # pytest sets PYTEST_CURRENT_TEST again before the test body runs
        for var in ("PYTEST_CURRENT_TEST", "CI", "QSEEK_TELEMETRY", "DO_NOT_TRACK"):
            monkeypatch.delenv(var, raising=False)

    return clear_environment


def fake_search(lat: float = 0.0, lon: float = 0.0) -> Any:
    return SimpleNamespace(
        created=datetime(2026, 1, 1, tzinfo=timezone.utc),
        stations=SimpleNamespace(n_stations=12, n_networks=2),
        octree=SimpleNamespace(
            n_nodes=1000,
            location=SimpleNamespace(effective_lat_lon=(lat, lon)),
        ),
        catalog=SimpleNamespace(n_events=5),
        ray_tracers=SimpleNamespace(
            root=[CakeTracer(), CakeTracer(), ConstantVelocityTracer()]
        ),
        station_corrections=None,
    )


def test_disabled_under_pytest():
    assert telemetry.disabled_by_environment()
    telemetry.init()
    assert telemetry._client is None


@pytest.mark.parametrize(
    ("var", "value"),
    [("QSEEK_TELEMETRY", "0"), ("DO_NOT_TRACK", "1"), ("CI", "true")],
)
def test_disabled_by_environment(telemetry_env, monkeypatch, var, value):
    telemetry_env()
    monkeypatch.setenv(var, value)
    telemetry.init()
    assert telemetry._client is None
    assert not telemetry.CONFIG_FILE.exists()


def test_notice_on_first_run_only(telemetry_env, capsys):
    telemetry_env()
    telemetry.init()
    assert "anonymous usage statistics" in capsys.readouterr().err
    assert telemetry.CONFIG_FILE.exists()
    assert isinstance(telemetry._client, FakePosthog)
    assert telemetry._client.kwargs["disable_geoip"] is False

    telemetry.init()
    assert capsys.readouterr().err == ""


def test_opt_out(telemetry_env):
    telemetry_env()
    telemetry.set_enabled(False)
    telemetry.init()
    assert telemetry._client is None


def test_run_events(telemetry_env):
    telemetry_env()
    telemetry.init()
    client = telemetry._client
    assert isinstance(client, FakePosthog)

    search = fake_search(lat=50.123456, lon=13.5)
    telemetry.capture_run_started(search, continued=False)
    stats = SimpleNamespace(processed_time=timedelta(days=2))
    telemetry.capture_run_finished(search, stats, timedelta(hours=1))  # type: ignore

    (started, started_kw), (finished, finished_kw) = client.events
    assert started == "run_started"
    assert finished == "run_finished"
    props = started_kw["properties"]
    assert props["$process_person_profile"] is False
    assert props["n_stations"] == 12
    assert props["ray_tracers"] == ["CakeTracer", "ConstantVelocityTracer"]
    assert props["station_corrections"] is None
    assert (props["octree_lat"], props["octree_lon"]) == (50.123, 13.5)
    assert props["run_id"] == finished_kw["properties"]["run_id"]
    assert finished_kw["properties"]["station_days"] == 24.0
    assert finished_kw["properties"]["n_detections"] == 5


def test_no_octree_center_at_origin(telemetry_env):
    telemetry_env()
    telemetry.init()
    client = telemetry._client
    assert isinstance(client, FakePosthog)
    telemetry.capture_run_started(fake_search(), continued=True)
    props = client.events[0][1]["properties"]
    assert "octree_lat" not in props
    assert props["continued"] is True


def test_library_use_sends_nothing():
    telemetry.capture_run_started(fake_search(), continued=False)
    assert telemetry._client is None
