from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import aiohttp
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from qseek.plugins.callback import Callback

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.search import Search

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramAlert(Callback, BaseSettings):
    """Sends detection alerts to a Telegram chat.

    `bot_token` is excluded from `search.json` so it never ends up on disk;
    set it (and optionally `chat_id`) through the QSEEK_TELEGRAM_BOT_TOKEN /
    QSEEK_TELEGRAM_CHAT_ID environment variables instead.
    """

    model_config = SettingsConfigDict(env_prefix="QSEEK_TELEGRAM_")

    callback: Literal["TelegramAlert"] = "TelegramAlert"

    bot_token: SecretStr = Field(
        default="xxx-xxx-xxx",
        description="The bot's token, as provided by BotFather.",
    )
    chat_id: str = Field(
        default="",
        description="The chat ID to send messages to."
        "If empty, the bot's default chat is used.",
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
                "parse_mode": "Markdown",
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
            await self._send(
                f"⚠️ *Swarm alert*: {n_events} events ≥ M{self.rate_alert_magnitude}"
                f" in the last {self.rate_alert_window}",
                latitude=detection.effective_lat,
                longitude=detection.effective_lon,
            )

    async def on_start(self, search: Search) -> None:
        await self._send(f"🚀 qseek search started: `{search.project_dir.name}`")

    async def on_new_detection(self, detection: EventDetection) -> None:
        magnitude = detection.magnitude
        if magnitude is None:
            return

        if self.magnitude_alert is None or magnitude.average >= self.magnitude_alert:
            await self._send(
                f"🔔 M{magnitude.average:.1f} {magnitude.name} detection",
                address=(
                    f"{detection.time.isoformat()} · "
                    f"{detection.effective_depth / 1e3:.1f} km depth"
                ),
                latitude=detection.effective_lat,
                longitude=detection.effective_lon,
            )

        if magnitude.average >= self.rate_alert_magnitude:
            await self._check_rate_alert(detection)

    async def on_stop(self, search: Search) -> None:
        await self._send(f"🏁 qseek search finished: `{search.project_dir.name}`")
