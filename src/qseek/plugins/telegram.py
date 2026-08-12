from __future__ import annotations

import html
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import aiohttp
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from qseek.magnitudes.base import EventMagnitude
from qseek.plugins.callback import Callback
from qseek.utils import datetime_pretty

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.search import Search

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


def get_magnitude_name(magnitude: EventMagnitude) -> str:
    if magnitude == "LocalMagnitude":
        return "ML"
    if magnitude == "MomentMagnitude":
        return "Mw"
    return magnitude.__class__.__name__


def _format_window(window: timedelta) -> str:
    hours = window.total_seconds() / 3600
    if hours > 24 and hours % 24 == 0:
        return f"{int(hours // 24)} days"
    if hours >= 1:
        return f"{hours:.0f} hours"
    return f"{window.total_seconds() / 60:.0f} minutes"


class TelegramAlert(Callback, BaseSettings):
    """Sends detection alerts to a Telegram chat.

    `bot_token` is excluded from `search.json` so it never ends up on disk;
    set it (and optionally `chat_id`) through the QSEEK_TELEGRAM_BOT_TOKEN /
    QSEEK_TELEGRAM_CHAT_ID environment variables instead.
    """

    model_config = SettingsConfigDict(env_prefix="QSEEK_TELEGRAM_")

    callback: Literal["TelegramAlert"] = "TelegramAlert"

    bot_token: SecretStr = Field(
        exclude=True,
        description="The bot's token, as provided by BotFather.",
    )
    chat_id: str = Field(
        description="The chat ID to send messages to.",
    )

    magnitude_alert: float | None = Field(
        default=2.0,
        description="Only notify for detections at or above this magnitude. "
        "Detections without a computed magnitude are not notified.",
    )

    rate_alert_magnitude: float = Field(
        default=1.0,
        description="Magnitude threshold considered for the rate alert.",
    )
    rate_alert_count: int = Field(
        default=10,
        description="Number of events at or above `rate_alert_magnitude` "
        "within `rate_alert_window` that triggers a swarm alert.",
    )
    rate_alert_window: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling time window for the rate alert.",
    )

    _recent_events: deque[datetime] = PrivateAttr(default_factory=deque)
    _rate_alerted: bool = PrivateAttr(False)
    _project_name: str = PrivateAttr("unknown")

    async def _send(
        self,
        text: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        address: str | None = None,
    ) -> None:
        # A venue pins the event on Telegram's native map card; a plain
        # message is used when no coordinates are given (start/stop/swarm).
        if latitude is not None and longitude is not None:
            method = "sendVenue"
            payload = {
                "chat_id": self.chat_id,
                "latitude": latitude,
                "longitude": longitude,
                "title": text,
                "address": address or f"{latitude:.4f}, {longitude:.4f}",
            }
        else:
            method = "sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }

        url = f"{TELEGRAM_API}/bot{self.bot_token.get_secret_value()}/{method}"
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(url, json=payload) as response,
        ):
            if response.status != 200:
                body = await response.text()
                logger.warning(
                    "telegram %s failed (%d): %s", method, response.status, body
                )

    async def _check_rate_alert(self, detection: EventDetection) -> None:
        self._recent_events.append(detection.time)
        cutoff = detection.time - self.rate_alert_window
        while self._recent_events and self._recent_events[0] < cutoff:
            self._recent_events.popleft()

        n_events = len(self._recent_events)
        if n_events < self.rate_alert_count:
            self._rate_alerted = False
            return

        if not self._rate_alerted:
            self._rate_alerted = True
            # sendVenue's title/address are always plain text, never parsed
            # as markup - keep it free of formatting syntax.
            await self._send(
                f"✨ Swarm alert: {n_events} events ≥ M{self.rate_alert_magnitude}"
                f" in the last {_format_window(self.rate_alert_window)}"
                f" · {self._project_name}",
                address=f"since {datetime_pretty(self._recent_events[0])}",
                latitude=detection.effective_lat,
                longitude=detection.effective_lon,
            )

    async def on_start(self, search: Search) -> None:
        self._project_name = search._rundir.name
        await self._send(
            f"🚀 qseek search started: <code>{html.escape(self._project_name)}</code>"
        )

    async def on_new_detection(self, detection: EventDetection) -> None:
        magnitude = detection.magnitude
        if magnitude is None:
            return

        if self.magnitude_alert is None or magnitude.average >= self.magnitude_alert:
            # sendVenue's title is always plain text, so the raw (unescaped)
            # project name is used here, unlike the HTML sendMessage calls.
            magnitude_name = get_magnitude_name(magnitude)
            await self._send(
                f"🎯 {magnitude_name}{magnitude.average:.1f} Event"
                f" · {self._project_name}",
                address=(
                    f"{datetime_pretty(detection.time)}\n"
                    f"{detection.effective_depth / 1e3:.1f} km depth"
                ),
                latitude=detection.effective_lat,
                longitude=detection.effective_lon,
            )

        if magnitude.average >= self.rate_alert_magnitude:
            await self._check_rate_alert(detection)

    async def on_stop(self, search: Search) -> None:
        name = html.escape(search.project_dir.name)
        await self._send(f"🏁 qseek search finished: <code>{name}</code>")
