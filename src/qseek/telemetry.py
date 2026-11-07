"""Anonymous usage statistics of searches, sent to PostHog.

The `qseek` command enables telemetry for `qseek search` and `qseek continue`.
Using Qseek as a Python library sends nothing. See the docs page
`about/telemetry.md` for the transmitted fields.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from datetime import timedelta

    from posthog import Posthog

    from qseek.search import Search, SearchStats

logger = logging.getLogger(__name__)

# Public project token, it can only send events
POSTHOG_API_KEY: str = "phc_tr3uq3qDyLQvdU7TYVJ6XR6dVUCuUDRsDxMPptRdADUD"
POSTHOG_HOST = "https://eu.i.posthog.com"

CONFIG_FILE = Path.home() / ".config" / "qseek" / "telemetry.json"
FLUSH_TIMEOUT = 2.0

NOTICE = f"""\
Qseek sends anonymous usage statistics of your searches: the Qseek, Python and
library versions, the operating system and hardware, the number of stations,
the center of the search volume, the processed data duration and the number of
detections. PostHog derives a coarse location from your IP address.
No station codes, waveforms, detections or file paths leave your machine.

Turn it off with `qseek telemetry off` or by setting QSEEK_TELEMETRY=0.
Details: https://pyrocko.github.io/qseek/about/telemetry/
This notice is shown once; the setting is stored in {CONFIG_FILE}.
"""

_FALSE_VALUES = {"0", "false", "off", "no"}

_client: Posthog | None = None
_install_id: UUID | None = None


class TelemetryConfig(BaseModel):
    enabled: bool = True
    install_id: UUID = Field(default_factory=uuid4)


def load_config() -> TelemetryConfig | None:
    """Load the telemetry config, None if it does not exist or is invalid."""
    try:
        return TelemetryConfig.model_validate_json(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        return None


def save_config(config: TelemetryConfig) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(config.model_dump_json(indent=2))


def disabled_by_environment() -> bool:
    """Telemetry is off in CI, under pytest and when the user opted out."""
    env = os.environ
    return (
        env.get("QSEEK_TELEMETRY", "").lower() in _FALSE_VALUES
        or env.get("DO_NOT_TRACK", "").lower() not in {"", *_FALSE_VALUES}
        or "CI" in env
        or "PYTEST_CURRENT_TEST" in env
    )


def set_enabled(enabled: bool) -> TelemetryConfig:
    config = load_config() or TelemetryConfig()
    config.enabled = enabled
    save_config(config)
    return config


def init() -> None:
    """Start the PostHog client; show the notice on the first run."""
    global _client, _install_id
    if not POSTHOG_API_KEY or disabled_by_environment():
        return

    config = load_config()
    if config is None:
        config = TelemetryConfig()
        try:
            save_config(config)
        except OSError:
            return  # Without a stored setting the notice would show on every run
        print(NOTICE, file=sys.stderr)  # noqa: T201
    if not config.enabled:
        return

    try:
        from posthog import Posthog

        _client = Posthog(
            POSTHOG_API_KEY,
            host=POSTHOG_HOST,
            disable_geoip=False,
            enable_exception_autocapture=False,
            max_retries=1,
            timeout=5,
        )
        # Network errors on offline machines must not reach the console. The
        # client sets the level of its logger, so this comes after it.
        logging.getLogger("posthog").setLevel(logging.CRITICAL)
    except Exception:
        logger.debug("could not start telemetry", exc_info=True)
        return
    _install_id = config.install_id


def capture(event: str, properties: dict[str, Any]) -> None:
    if _client is None or _install_id is None:
        return
    try:
        _client.capture(
            event,
            distinct_id=str(_install_id),
            properties={"$process_person_profile": False, **properties},
        )
    except Exception:
        logger.debug("could not capture telemetry event", exc_info=True)


def flush() -> None:
    """Send queued events, waiting at most `FLUSH_TIMEOUT` seconds."""
    if _client is None:
        return
    try:
        _client.flush(timeout_seconds=FLUSH_TIMEOUT)
    except Exception:
        logger.debug("could not flush telemetry", exc_info=True)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _gpu() -> str:
    torch = sys.modules.get("torch")  # Loaded by the image function, if any
    if torch is None:
        return "none"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "none"


def run_id(search: Search) -> str:
    """ID of a run, the same when the run is continued."""
    return str(uuid5(_install_id or uuid4(), search.created.isoformat()))


def capture_run_started(search: Search, continued: bool) -> None:
    if _client is None:
        return
    properties: dict[str, Any] = {
        "run_id": run_id(search),
        "continued": continued,
        "qseek_version": _package_version("qseek"),
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "seisbench_version": _package_version("seisbench"),
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu(),
        "n_stations": search.stations.n_stations,
        "n_networks": search.stations.n_networks,
        "n_nodes": search.octree.n_nodes,
    }
    lat, lon = search.octree.location.effective_lat_lon
    if lat != 0.0 or lon != 0.0:
        properties["octree_lat"] = round(lat, 3)
        properties["octree_lon"] = round(lon, 3)
    capture("run_started", properties)


def capture_run_finished(
    search: Search, stats: SearchStats, wall_time: timedelta
) -> None:
    if _client is None:
        return
    processed_seconds = stats.processed_time.total_seconds()
    capture(
        "run_finished",
        {
            "run_id": run_id(search),
            "wall_time_seconds": round(wall_time.total_seconds()),
            "processed_seconds": round(processed_seconds),
            "station_days": round(
                processed_seconds * search.stations.n_stations / 86400, 2
            ),
            "n_detections": search.catalog.n_events,
        },
    )
