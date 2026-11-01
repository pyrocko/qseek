from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qseek.models.detection import EventDetection, EventReceivers


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


def test_receivers_stored_as_list() -> None:
    """Receivers of runs are a list, the lookups go by NSL."""
    stored = (
        '{"event_uid":null,"receivers":['
        '{"network":"XX","station":"STA","location":"","lat":40.8,"lon":14.1},'
        '{"network":"XX","station":"STB","location":"00","lat":40.9,"lon":14.2}'
        "]}"
    )
    receivers = EventReceivers.model_validate_json(stored)
    assert receivers.n_receivers == 2
    assert receivers.get_receiver(("XX", "STB", "00")).lat == 40.9
    assert [rcv.station for rcv in receivers] == ["STA", "STB"]
    with pytest.raises(KeyError, match=r"XX\.STC\."):
        receivers.get_receiver(("XX", "STC", ""))

    dumped = receivers.model_dump(mode="json")["receivers"]
    assert [rcv["station"] for rcv in dumped] == ["STA", "STB"]
    assert EventReceivers.model_validate_json(receivers.model_dump_json()) == receivers
