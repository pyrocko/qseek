from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.features.ground_motion import EventGroundMotion, GroundMotionExtractor
from qseek.models.detection import EventDetection, EventReceivers, Receiver

PEAKS = {
    # station: (acceleration E, N, Z), (velocity E, N, Z)
    "STA1": ((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
    "STA2": ((4.0, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def _traces(station: str, quantity: str) -> list[Trace]:
    peaks = PEAKS[station][0 if quantity == "acceleration" else 1]
    traces = []
    for channel, peak in zip("ENZ", peaks, strict=True):
        data = np.zeros(100)
        data[50] = peak
        traces.append(Trace("XX", station, "", f"HH{channel}", deltat=0.01, ydata=data))
    return traces


@pytest.mark.asyncio
async def test_ground_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_waveforms_restituted(
        self, waveform_provider, stations, receivers, quantity, **kwargs
    ) -> list[Trace]:
        (receiver,) = receivers
        return _traces(receiver.station, quantity)

    monkeypatch.setattr(
        EventReceivers, "get_waveforms_restituted", get_waveforms_restituted
    )
    detection = EventDetection(
        lat=40.8,
        lon=14.1,
        time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        semblance=0.5,
        distance_border=1000.0,
    )
    detection.receivers = EventReceivers(
        event_uid=detection.uid,
        receivers=[
            Receiver(network="XX", station=station, lat=40.8, lon=14.1)
            for station in PEAKS
        ],
    )

    await GroundMotionExtractor().add_features(None, None, detection)

    (feature,) = detection.features
    assert isinstance(feature, EventGroundMotion)
    assert feature.peak_ground_acceleration == pytest.approx(np.sqrt(4.0**2 + 0.5))
    assert feature.peak_horizontal_acceleration == pytest.approx(np.sqrt(4.0**2 + 0.25))
    assert feature.peak_ground_velocity == pytest.approx(np.sqrt(0.75))
