"""Live smoke tests for TelegramAlert - these actually send messages.

Requires QSEEK_TELEGRAM_BOT_TOKEN / QSEEK_TELEGRAM_CHAT_ID, loaded from a
gitignored `.env` in the project root if present. Skipped automatically
when unset, so this never runs in CI or on a machine without credentials.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

from qseek.magnitudes.local_magnitude import EventLocalMagnitude
from qseek.models.detection import EventDetection
from qseek.plugins.telegram import TelegramAlert

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("QSEEK_TELEGRAM_BOT_TOKEN"),
    reason="requires QSEEK_TELEGRAM_BOT_TOKEN / QSEEK_TELEGRAM_CHAT_ID",
)


def make_detection(magnitude: float, minutes: int = 0) -> EventDetection:
    detection = EventDetection(
        lat=52.5200,
        lon=13.4050,
        depth=5300.0,
        time=datetime.now(tz=timezone.utc) + timedelta(minutes=minutes),
        semblance=0.85,
        distance_border=5000.0,
    )
    detection.add_magnitude(EventLocalMagnitude(average=magnitude))
    return detection


@pytest.fixture
def alert() -> TelegramAlert:
    return TelegramAlert(rate_alert_count=3)


@pytest.fixture
def search() -> SimpleNamespace:
    return SimpleNamespace(_rundir=Path("test-run"))


@pytest.mark.asyncio
async def test_telegram_on_start(alert: TelegramAlert, search: SimpleNamespace):
    await alert.on_start(search)


@pytest.mark.asyncio
async def test_telegram_new_detection(alert: TelegramAlert):
    await alert.on_new_detection(make_detection(magnitude=2.3))


@pytest.mark.asyncio
async def test_telegram_magnitude_alert_filters(alert: TelegramAlert):
    alert.magnitude_alert = 3.0
    await alert.on_new_detection(make_detection(magnitude=1.5))  # not sent


@pytest.mark.asyncio
async def test_telegram_swarm_alert(alert: TelegramAlert):
    for i in range(alert.rate_alert_count):
        await alert.on_new_detection(make_detection(magnitude=1.1, minutes=i))


@pytest.mark.asyncio
async def test_telegram_on_stop(alert: TelegramAlert, search: SimpleNamespace):
    await alert.on_stop(search)
