from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream
from pydantic import ValidationError
from pyrocko.trace import Trace

from qseek.images.seisbench import AnnotationPicker, SeisBench

TMIN = 1700000000.123


def time(seconds: float) -> datetime:
    return datetime.fromtimestamp(TMIN + seconds, tz=timezone.utc)


def annotation_trace(
    peaks: dict[float, float],
    sampling_rate: float = 100.0,
    duration: float = 10.0,
    phase: str = "P",
    station: str = "STA",
) -> Trace:
    """Annotation trace starting at `TMIN` with peaks at {seconds: value}."""
    data = np.zeros(round(duration * sampling_rate))
    for seconds, value in peaks.items():
        data[round(seconds * sampling_rate)] = value
    return Trace(
        network="XX",
        station=station,
        channel=phase,
        tmin=TMIN,
        deltat=1 / sampling_rate,
        ydata=data,
    )


def pick_one(
    picker: AnnotationPicker, trace: Trace, modelled_arrival: datetime, event_time=None
):
    return picker.pick_trace(trace, "cake:P", event_time or time(0), modelled_arrival)


@pytest.mark.parametrize("sampling_rate", [5, 25, 100, 200])
def test_nearest_peak_time_after_chopping(sampling_rate):
    trace = annotation_trace({5.0: 0.35, 5.6: 0.9}, sampling_rate=sampling_rate)
    picker = AnnotationPicker()

    # The window starts between samples and the stronger peak is less than 1 s away.
    pick = pick_one(picker, trace, time(5.003))
    assert pick is not None
    assert pick.time == time(5)
    assert pick.detection_value == 0.35
    assert pick.phase == "cake:P"

    pick = pick_one(picker, trace, time(5.5))
    assert pick is not None
    assert pick.time == time(5.6)

    pick = pick_one(AnnotationPicker(threshold_p=0.5), trace, time(5))
    assert pick is not None
    assert pick.time == time(5.6)


@pytest.mark.parametrize(
    "separation_seconds,expected_seconds", [(None, 5.06), (0.02, 5.0), (0.0, 5.0)]
)
def test_peak_separation(separation_seconds, expected_seconds):
    trace = annotation_trace({5.0: 0.35, 5.06: 0.9})
    picker = (
        AnnotationPicker()
        if separation_seconds is None
        else AnnotationPicker(peak_separation_seconds=separation_seconds)
    )
    pick = pick_one(picker, trace, time(5))
    assert pick is not None
    assert pick.time == time(expected_seconds)


@pytest.mark.parametrize(
    "phase,threshold_p,threshold_s,expected_seconds",
    [
        ("P", 0.1, 0.5, 5.0),
        ("P", 0.5, 0.1, 5.6),
        ("S", 0.1, 0.5, 5.6),
        ("S", 0.5, 0.1, 5.0),
    ],
)
def test_phase_threshold(phase, threshold_p, threshold_s, expected_seconds):
    trace = annotation_trace({5.0: 0.35, 5.6: 0.9}, phase=phase)
    picker = AnnotationPicker(threshold_p=threshold_p, threshold_s=threshold_s)
    pick = pick_one(picker, trace, time(5))
    assert pick is not None
    assert pick.time == time(expected_seconds)


def test_phase_threshold_unknown_channel():
    trace = annotation_trace({5.0: 0.9}, phase="HHZ")
    with pytest.raises(ValueError, match="No pick threshold"):
        pick_one(AnnotationPicker(), trace, time(5))


def test_reject_pre_event_peaks():
    trace = annotation_trace({4.0: 0.9, 5.5: 0.3})
    picker = AnnotationPicker()

    pick = pick_one(picker, trace, time(4.2), event_time=time(3))
    assert pick is not None
    assert pick.time == time(4)

    # The closest peak precedes the event, the later one is picked
    pick = pick_one(picker, trace, time(4.2), event_time=time(4.5))
    assert pick is not None
    assert pick.time == time(5.5)

    pick = pick_one(picker, trace, time(6), event_time=time(6))
    assert pick is None


def test_search_window():
    trace = annotation_trace({2.0: 0.9}, duration=20.0)
    assert pick_one(AnnotationPicker(), trace, time(5)) is None

    pick = pick_one(AnnotationPicker(search_window_seconds=8.0), trace, time(5))
    assert pick is not None
    assert pick.time == time(2)


def test_no_data():
    trace = annotation_trace({5.0: 0.9})
    assert pick_one(AnnotationPicker(), trace, time(60)) is None


def test_picker_config():
    function = SeisBench.model_validate(
        {"picker": {"threshold_p": 0.3, "search_window_seconds": 2.0}}
    )
    assert function.picker.threshold_p == 0.3
    assert function.picker.threshold_s == 0.1
    assert function.picker.search_window_seconds == 2.0

    with pytest.raises(ValidationError):
        SeisBench.model_validate({"rescale_input": 1.0})

    loaded = SeisBench.model_validate_json(function.model_dump_json())
    assert loaded.picker == function.picker

    for invalid in (
        {"threshold_p": 0.0},
        {"threshold_s": 1.5},
        {"search_window_seconds": 0.0},
        {"peak_separation_seconds": -1.0},
        {"unknown": 1.0},
        {"image": "SeisBench"},
    ):
        with pytest.raises(ValidationError):
            AnnotationPicker.model_validate(invalid)


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
    """Restore annotation offsets without depending on downloaded model weights.

    The pre-trained model's sampling rate is set to the input sampling rate, the
    model annotates the input without resampling.
    """
    tmin = 1700000000.123
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
            assert trace.stats.sampling_rate == sampling_rate
            assert trace.stats.starttime.timestamp == pytest.approx(
                original.tmin, rel=0, abs=1e-6
            )
            for phase in ("P", "S"):
                annotation = trace.copy()
                annotation.data = trace.data[
                    leading_samples : -trailing_samples or None : stride
                ].copy()
                annotation.stats.starttime += leading_samples / sampling_rate
                annotation.stats.sampling_rate = sampling_rate / stride
                annotation.stats.channel = f"{model}_{phase}"
                annotations.append(annotation)
        return annotations

    # Keep this deterministic conversion test independent of thread scheduling.
    async def inline_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("qseek.images.seisbench.asyncio.to_thread", inline_thread)
    function = SeisBench(model=model, sampling_rate=sampling_rate)
    function._seisbench_model = SimpleNamespace(
        annotate=annotate, sampling_rate=sampling_rate
    )

    images = await function.process_traces(traces)
    for image in images:
        expected_arrivals = [
            datetime.fromtimestamp(original.tmin + 20, tz=timezone.utc)
            for original in traces
        ]
        picks = [
            function.picker.pick_trace(
                annotation,
                image.phase,
                datetime.fromtimestamp(tmin, tz=timezone.utc),
                expected,
            )
            for annotation, expected in zip(
                image.traces, expected_arrivals, strict=True
            )
        ]
        for original, annotation, pick, expected in zip(
            traces, image.traces, picks, expected_arrivals, strict=True
        ):
            assert annotation.tmin == pytest.approx(
                original.tmin + leading_samples / sampling_rate, rel=0, abs=1e-6
            )
            assert annotation.deltat == pytest.approx(stride / sampling_rate)
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


@pytest.mark.asyncio
async def test_sampling_rate_input(monkeypatch):
    """`sampling_rate="input"` follows the traces and rejects mixed rates."""

    def trace(sampling_rate: float) -> Trace:
        return Trace(
            network="XX",
            station="STA",
            channel="HHZ",
            tmin=TMIN,
            deltat=1 / sampling_rate,
            ydata=np.zeros(int(10 * sampling_rate)),
        )

    async def inline_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("qseek.images.seisbench.asyncio.to_thread", inline_thread)
    function = SeisBench(sampling_rate="input")
    function._seisbench_model = SimpleNamespace(
        annotate=lambda stream, **kwargs: Stream(),
        sampling_rate=100.0,
        default_args={"blinding": (100, 300)},
    )
    function._native_sampling_rate = 100.0

    await function.process_traces([trace(200.0)])
    assert function._seisbench_model.sampling_rate == 200.0
    assert function._rescale_input == 2.0

    with pytest.raises(ValueError, match="homogeneous"):
        await function.process_traces([trace(100.0), trace(200.0)])


@pytest.mark.asyncio
async def test_sampling_rate_input_warns_on_longer_blinding(monkeypatch, caplog):
    """A lower input rate than the padded one lengthens the blinding."""

    async def inline_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("qseek.images.seisbench.asyncio.to_thread", inline_thread)
    function = SeisBench(sampling_rate="input")
    function._seisbench_model = SimpleNamespace(
        annotate=lambda stream, **kwargs: Stream(),
        sampling_rate=100.0,
        default_args={"blinding": (100, 300)},
    )
    function._native_sampling_rate = 100.0
    assert function.get_blinding().total_seconds() == 3.0

    trace = Trace(
        network="XX",
        station="STA",
        channel="HHZ",
        tmin=TMIN,
        deltat=1 / 50.0,
        ydata=np.zeros(500),
    )
    with caplog.at_level("WARNING"):
        await function.process_traces([trace])
    assert "longer than" in caplog.text
