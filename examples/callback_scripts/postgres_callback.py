"""Insert new detections directly into the qseekv2 Postgres database.

Enable by adding this file's path to the search config:

    "callback_scripts": ["examples/callback_scripts/postgres_callback.py"]

Connects as the `qseekv2_producer` role, authenticated via `~/.pgpass`
(override with the QSEEKV2_DATABASE / QSEEKV2_PRODUCER_USER environment
variables). Inserts go through the `forge.add_event` stored procedure, which
is idempotent on (time, latitude, longitude, depth).

Run this file directly (`python postgres_callback.py`) to check that the
connection and role are set up correctly, without running a search.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone
from typing import TYPE_CHECKING

import psycopg
from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from qseek.plugins.callback import Callback

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection

logger = logging.getLogger(__name__)

ALGORITHM = "qseek"


class PostgresCallback(Callback, BaseSettings):
    """Forwards new detections to the qseekv2 Postgres database.

    Connection settings (`database`, `producer_user`) are read from the
    QSEEKV2_* environment, falling back to the defaults below.
    """

    model_config = SettingsConfigDict(env_prefix="QSEEKV2_")

    database: str = "qseekv2"
    producer_user: str = "qseekv2_producer"

    _connection: psycopg.AsyncConnection | None = PrivateAttr(None)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def _connect(self) -> psycopg.AsyncConnection:
        if self._connection is None or self._connection.closed:
            self._connection = await psycopg.AsyncConnection.connect(
                dbname=self.database,
                user=self.producer_user,
                autocommit=True,
            )
        return self._connection

    async def _insert(self, detection: EventDetection) -> int:
        magnitude = detection.magnitude
        # The column is TIMESTAMP WITHOUT TIME ZONE holding UTC. Handing
        # psycopg a timezone-aware value would have the server convert it
        # using *its* timezone setting, silently storing the wrong instant.
        time = detection.time.astimezone(timezone.utc).replace(tzinfo=None)

        # A connection handles one statement at a time, and on_new_detection
        # can be entered concurrently for detections from the same batch.
        async with self._lock:
            connection = await self._connect()
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT forge.add_event(%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        time,
                        detection.effective_lat,
                        detection.effective_lon,
                        detection.effective_depth,
                        magnitude.average if magnitude else None,
                        magnitude.name if magnitude else None,
                        ALGORITHM,
                        True,
                    ),
                )
                row = await cursor.fetchone()
                return row[0]

    async def on_new_detection(self, detection: EventDetection) -> None:
        try:
            identifier = await self._insert(detection)
        except psycopg.OperationalError:
            # Connection died between calls - server bounced, network
            # blipped. Drop it and retry once; add_event is idempotent so a
            # retry after a successful-but-unacknowledged insert is safe.
            async with self._lock:
                if self._connection is not None:
                    await self._connection.close()
                self._connection = None
            identifier = await self._insert(detection)

        logger.info(
            "added detection %s to qseekv2 as event %d", detection.uid, identifier
        )


def load() -> Callback:
    return PostgresCallback()


async def test_connection() -> None:
    """Connect as PostgresCallback would and run a trivial round trip."""
    callback = PostgresCallback()
    connection = await callback._connect()
    async with connection.cursor() as cursor:
        await cursor.execute("SELECT 1")
        row = await cursor.fetchone()
    await connection.close()
    assert row == (1,), f"unexpected result from qseekv2: {row}"
    logger.info("connected to qseekv2 as %s", callback.producer_user)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_connection())
