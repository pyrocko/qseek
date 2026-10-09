from __future__ import annotations

from datetime import datetime, timezone

from qseek.models.detection import EventDetection


def test_receivers_without_index() -> None:
    """A detection that is not saved in a catalog has no receivers."""
    detection = EventDetection(
        lat=40.8,
        lon=14.1,
        time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        semblance=0.5,
        distance_border=1000.0,
    )
    receivers = detection.receivers
    assert receivers.n_receivers == 0
    assert receivers.event_uid == detection.uid
    assert detection.receivers is receivers
