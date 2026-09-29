from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream
from pyrocko.trace import Trace

from qseek.images.seisbench import PhaseNetImage, SeisBench


@pytest.mark.parametrize("sampling_rate", [5, 25, 100, 200])
def test_nearest_peak_time_after_chopping(sampling_rate):
    tmin = 1700000000.123
    data = np.zeros(10 * sampling_rate)
    data[5 * sampling_rate] = 0.35
    data[round(5.6 * sampling_rate)] = 0.9
    trace = Trace(tmin=tmin, deltat=1 / sampling_rate, ydata=data)
    image = PhaseNetImage("SeisBench", "cake:P", 1.0, [trace], 0.2)

    def time(seconds):
        return datetime.fromtimestamp(tmin + seconds, tz=timezone.utc)

    # The window starts between samples and the stronger peak is less than 1 s away.
    pick = image.search_phase_arrival(0, time(0), time(5.003))
    assert pick is not None
    assert pick.time == time(5)
    assert pick.detection_value == 0.35

    pick = image.search_phase_arrival(0, time(0), time(5.5))
    assert pick is not None
    assert pick.time == time(5.6)

    pick = image.search_phase_arrival(0, time(0), time(5), threshold=0.5)
    assert pick is not None
    assert pick.time == time(5.6)


@pytest.mark.parametrize(
    "blinding_seconds,expected_seconds", [(None, 5.06), (0.02, 5.0), (0.0, 5.0)]
)
def test_peak_blinding(blinding_seconds, expected_seconds):
    tmin = 1700000000.123
    data = np.zeros(1000)
    data[500], data[506] = 0.35, 0.9
    image = PhaseNetImage(
        "SeisBench", "cake:P", 1.0, [Trace(tmin=tmin, deltat=0.01, ydata=data)], 0.2
    )
    kwargs = (
        {}
        if blinding_seconds is None
        else {"detection_blinding_seconds": blinding_seconds}
    )
    pick = image.search_phase_arrival(
        0,
        datetime.fromtimestamp(tmin, tz=timezone.utc),
        datetime.fromtimestamp(tmin + 5, tz=timezone.utc),
        **kwargs,
    )
    assert pick is not None
    assert pick.time == datetime.fromtimestamp(tmin + expected_seconds, tz=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("sampling_rate", [50, 100, 200])
@pytest.mark.parametrize(
    "model,leading_samples,trailing_samples,stride",
    [
        ("PhaseNet", 0, 0, 1),
        ("PhaseNet", 100, 300, 1),
        ("EQTransformer", 500, 500, 1),
        ("GPD", 200, 200, 10),
    ],
)
async def test_annotation_sample_times(
    monkeypatch, sampling_rate, model, leading_samples, trailing_samples, stride
):
    """Restore annotation offsets without depending on downloaded model weights."""
    tmin = 1700000000.123
    scale = sampling_rate / 100
    traces = []
    # Include different component starts, a gap, and a second station. The earliest
    # trace is deliberately not first in the stream.
    for station, channel, offset in [
        ("STA", "HHN", 3.2),
        ("STA", "HHZ", 0),
        ("STA", "HHZ", 50),
        ("STB", "HHZ", 2),
    ]:
        data = np.zeros(40 * sampling_rate)
        data[20 * sampling_rate] = 0.9
        traces.append(
            Trace(
                network="XX",
                station=station,
                channel=channel,
                tmin=tmin + offset,
                deltat=1 / sampling_rate,
                ydata=data,
            )
        )

    def annotate(stream, **kwargs):
        annotations = Stream()
        for original, trace in zip(traces, stream, strict=True):
            assert trace.stats.sampling_rate == 100
            assert trace.stats.starttime.timestamp == pytest.approx(
                tmin + (original.tmin - tmin) * scale, rel=0, abs=1e-6
            )
            for phase in ("P", "S"):
                annotation = trace.copy()
                annotation.data = trace.data[
                    leading_samples : -trailing_samples or None : stride
                ].copy()
                annotation.stats.starttime += leading_samples / 100
                annotation.stats.sampling_rate = 100 / stride
                annotation.stats.channel = f"{model}_{phase}"
                annotations.append(annotation)
        return annotations

    # Keep this deterministic conversion test independent of thread scheduling.
    async def inline_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("qseek.images.seisbench.asyncio.to_thread", inline_thread)
    function = SeisBench(model=model, sampling_rate=sampling_rate)
    function._rescale_input = scale
    function._seisbench_model = SimpleNamespace(annotate=annotate, sampling_rate=100)

    images = await function.process_traces(traces)
    for image in images:
        for idx, (original, annotation) in enumerate(
            zip(traces, image.traces, strict=True)
        ):
            assert annotation.tmin == pytest.approx(
                original.tmin + leading_samples / sampling_rate, rel=0, abs=1e-6
            )
            assert annotation.deltat == pytest.approx(stride / sampling_rate)
            expected = datetime.fromtimestamp(original.tmin + 20, tz=timezone.utc)
            pick = image.search_phase_arrival(
                idx,
                datetime.fromtimestamp(original.tmin, tz=timezone.utc),
                expected,
            )
            assert pick is not None
            assert abs((pick.time - expected).total_seconds()) <= 1e-6
            assert original.deltat == 1 / sampling_rate
            assert original.ydata.argmax() == 20 * sampling_rate


@pytest.mark.parametrize("sampling_rate", [50, 100, 200])
def test_blinding_duration(sampling_rate):
    function = SeisBench(sampling_rate=sampling_rate)
    function._rescale_input = sampling_rate / 100
    function._seisbench_model = SimpleNamespace(default_args={"blinding": (100, 300)})
    assert function.get_blinding().total_seconds() == 300 / sampling_rate
