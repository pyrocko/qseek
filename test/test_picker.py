from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.images.base import (
    ObservedArrival,
    Picker,
    WaveformImage,
    WaveformImages,
)
from qseek.models.detection import EventDetection, PhaseDetection
from qseek.models.station import Station
from qseek.tracers.base import ModelledArrival

TMIN = 1700000000.0
SAMPLING_RATE = 10.0


def time(seconds: float) -> datetime:
    return datetime.fromtimestamp(TMIN + seconds, tz=timezone.utc)


class MaxPicker(Picker):
    """Picks the maximum within one second around the modelled arrival."""

    def pick_trace(self, trace, phase, event_time, modelled_arrival):
        times = trace.get_xdata()
        window = np.abs(times - modelled_arrival.timestamp()) <= 1.0
        if not window.any():
            return None
        idx = np.flatnonzero(window)[trace.ydata[window].argmax()]
        if trace.ydata[idx] <= 0.0:
            return None
        return ObservedArrival(
            phase=phase,
            time=datetime.fromtimestamp(times[idx], tz=timezone.utc),
            detection_value=float(trace.ydata[idx]),
        )


def station(name: str) -> Station:
    return Station(network="XX", station=name, location="", lat=0.0, lon=0.0)


def image_trace(
    name: str, peaks: dict[float, float], tmin: float = 0.0, duration: float = 30.0
) -> Trace:
    data = np.zeros(round(duration * SAMPLING_RATE))
    for seconds, value in peaks.items():
        data[round((seconds - tmin) * SAMPLING_RATE)] = value
    return Trace("XX", name, "", "Z", TMIN + tmin, deltat=1 / SAMPLING_RATE, ydata=data)


def waveform_image(phase: str, traces: list[Trace]) -> WaveformImage:
    return WaveformImage("Test", phase, 1.0, traces, 0.2)


def detection(
    modelled: dict[str, dict[str, float | None]], event_time: float = 0.0
) -> EventDetection:
    """Detection with modelled arrivals {phase: {station: seconds | None}}."""
    event = EventDetection(
        time=time(event_time),
        semblance=1.0,
        distance_border=1.0,
        lat=0.0,
        lon=0.0,
    )
    for phase, arrivals in modelled.items():
        event.receivers.add(
            stations=[station(name) for name in arrivals],
            phase_arrivals=[
                PhaseDetection(
                    phase=phase,
                    model=ModelledArrival(phase=phase, time=time(seconds)),
                )
                if seconds is not None
                else None
                for seconds in arrivals.values()
            ],
        )
    return event


def observed(event: EventDetection, name: str, phase: str) -> ObservedArrival | None:
    receiver = event.receivers.get_by_nsl(station(name).nsl)
    return receiver.phase_arrivals[phase].observed


def test_add_picks():
    images = WaveformImages(start_time=time(0), end_time=time(30))
    # The station sets differ between the phase images
    images.add_image(
        waveform_image(
            "cake:P",
            [image_trace("STA", {5.0: 0.8}), image_trace("STB", {6.0: 0.7})],
        )
    )
    images.add_image(
        waveform_image(
            "cake:S",
            [image_trace("STB", {10.0: 0.6}), image_trace("STC", {11.0: 0.5})],
        )
    )
    event = detection(
        {
            "cake:P": {"STA": 5.2, "STB": 6.1, "STC": 7.0},
            "cake:S": {"STB": 9.8, "STC": 11.3, "STA": None},
        }
    )

    MaxPicker().add_picks([event], images)

    pick = observed(event, "STA", "cake:P")
    assert pick is not None
    assert pick.time == time(5.0)
    assert pick.phase == "cake:P"
    assert pick.detection_value == 0.8

    pick = observed(event, "STB", "cake:P")
    assert pick is not None
    assert pick.time == time(6.0)

    pick = observed(event, "STB", "cake:S")
    assert pick is not None
    assert pick.time == time(10.0)
    assert pick.phase == "cake:S"

    pick = observed(event, "STC", "cake:S")
    assert pick is not None
    assert pick.time == time(11.0)

    # STC has no P image trace
    assert observed(event, "STC", "cake:P") is None
    # STA has no modelled S arrival
    sta = event.receivers.get_by_nsl(station("STA").nsl)
    assert "cake:S" not in sta.phase_arrivals


def test_add_picks_multiple_detections():
    images = WaveformImages(start_time=time(0), end_time=time(30))
    images.add_image(
        waveform_image("cake:P", [image_trace("STA", {5.0: 0.8, 20.0: 0.9})])
    )
    first = detection({"cake:P": {"STA": 5.3}}, event_time=2.0)
    second = detection({"cake:P": {"STA": 19.6}}, event_time=17.0)

    MaxPicker().add_picks([first, second], images)

    assert observed(first, "STA", "cake:P").time == time(5.0)
    assert observed(second, "STA", "cake:P").time == time(20.0)


def test_add_picks_no_pick():
    images = WaveformImages(start_time=time(0), end_time=time(30))
    images.add_image(waveform_image("cake:P", [image_trace("STA", {})]))
    event = detection({"cake:P": {"STA": 5.0}})

    MaxPicker().add_picks([event], images)

    assert observed(event, "STA", "cake:P") is None


def test_add_picks_data_gap():
    """A station split by a gap is picked on the trace covering the arrival."""
    images = WaveformImages(start_time=time(0), end_time=time(30))
    images.add_image(
        waveform_image(
            "cake:P",
            [
                image_trace("STA", {}, tmin=0.0, duration=10.0),
                image_trace("STA", {15.0: 0.8}, tmin=12.0, duration=18.0),
            ],
        )
    )
    event = detection({"cake:P": {"STA": 15.3}})

    MaxPicker().add_picks([event], images)

    pick = observed(event, "STA", "cake:P")
    assert pick is not None
    assert pick.time == time(15.0)


def test_add_picks_modelled_arrival_with_station_delay():
    """The modelled arrival includes the station delay, picks search around it."""
    images = WaveformImages(start_time=time(0), end_time=time(30))
    images.add_image(
        waveform_image("cake:P", [image_trace("STA", {5.0: 0.8, 8.0: 0.9})])
    )
    event = detection({"cake:P": {"STA": 5.0}})
    arrival = event.receivers.get_by_nsl(station("STA").nsl).phase_arrivals["cake:P"]
    arrival.station_delay = timedelta(seconds=3.0)
    arrival.model.time += arrival.station_delay

    MaxPicker().add_picks([event], images)

    assert observed(event, "STA", "cake:P").time == time(8.0)


def test_images_mixed_sampling_rates():
    traces = [image_trace("STA", {}), image_trace("STB", {})]
    traces[1].resample(0.05)
    images = WaveformImages(start_time=time(0), end_time=time(30))
    with pytest.raises(ValueError, match="different sampling rates"):
        images.add_image(waveform_image("cake:P", traces))
