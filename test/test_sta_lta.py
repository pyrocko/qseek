from datetime import datetime, timezone

import numpy as np
import pytest
from obspy import Stream
from pydantic import ValidationError
from pyrocko.obspy_compat import to_pyrocko_traces
from pyrocko.trace import Trace

from qseek.images.sta_lta import (
    StaLta,
    StaLtaPicker,
    _centred_sta_lta,
    _compute_characteristic_functions,
    _log_onset,
    _merge_horizontal_components,
    _overlapping_sta_lta,
)

TMIN = 1700000000.123
SAMPLING_RATE = 100.0


def time(seconds: float) -> datetime:
    return datetime.fromtimestamp(TMIN + seconds, tz=timezone.utc)


def sta_lta_trace(
    triggers: dict[tuple[float, float], float],
    duration: float = 60.0,
    station: str = "STA",
    seed: int = 0,
) -> Trace:
    """STA/LTA-like trace with a noisy baseline at 1 and {(start, end): value}."""
    rng = np.random.default_rng(seed)
    data = 1.0 + 0.05 * rng.standard_normal(round(duration * SAMPLING_RATE))
    for (start, end), value in triggers.items():
        data[round(start * SAMPLING_RATE) : round(end * SAMPLING_RATE)] = value
    return Trace(
        network="XX",
        station=station,
        channel="Z",
        tmin=TMIN,
        deltat=1 / SAMPLING_RATE,
        ydata=data,
    )


def pick_one(
    picker: StaLtaPicker, trace: Trace, modelled_arrival: datetime, event_time=None
):
    return picker.pick_trace(trace, "cake:P", event_time or time(0), modelled_arrival)


def test_trigger_thresholds():
    data = np.array([1.0, 1.0, 1.1, 0.9, 1.2, 0.8, 5.0])
    threshold_on, threshold_off = StaLtaPicker(mad_factor=4.0).get_trigger_thresholds(
        data
    )
    # median = 1.0, MAD = 0.1
    assert threshold_on == pytest.approx(1.4)
    assert threshold_off == pytest.approx(1.2)


def test_pick_onset():
    trace = sta_lta_trace({(30.0, 31.0): 3.0})
    pick = pick_one(StaLtaPicker(), trace, time(30.5))
    assert pick is not None
    assert pick.time == time(30.0)
    assert pick.phase == "cake:P"


def test_pick_trigger_peak():
    trace = sta_lta_trace({(30.0, 30.5): 2.0, (30.5, 31.0): 4.0})
    pick = pick_one(StaLtaPicker(), trace, time(30.0))
    assert pick is not None
    assert pick.time == time(30.5)
    assert pick.detection_value == 4.0


def test_centred_sta_lta():
    signal = np.ones(1000)
    signal[600:] = 100.0
    ratio = _centred_sta_lta(signal, nsta=20, nlta=100)

    # Peaks one sample before the onset, the STA window starts at the onset
    assert ratio.argmax() == 599
    assert ratio.max() == pytest.approx(100.0)
    # Null result where the LTA or STA window is not filled
    np.testing.assert_array_equal(ratio[:99], 1.0)
    np.testing.assert_array_equal(ratio[-20:], 1.0)


def test_centred_sta_lta_windows():
    rng = np.random.default_rng(0)
    signal = rng.random(500) ** 2
    nsta, nlta = 7, 30
    ratio = _centred_sta_lta(signal, nsta, nlta)

    for idx in range(nlta - 1, signal.size - nsta):
        sta = signal[idx + 1 : idx + 1 + nsta].mean()
        lta = signal[idx - nlta + 1 : idx + 1].mean()
        assert ratio[idx] == pytest.approx(sta / lta)


def test_classic_sta_lta():
    signal = np.ones(1000)
    signal[600:] = 100.0
    ratio = _overlapping_sta_lta(signal, nsta=20, nlta=100)

    # Peaks when the STA window is filled, limited by the overlapping LTA window
    assert ratio.argmax() == 619
    assert ratio.max() <= 100 / 20


def test_no_pick_in_noise():
    """The threshold is relative to the baseline, noise does not trigger."""
    trace = sta_lta_trace({(30.0, 31.0): 3.0})
    assert pick_one(StaLtaPicker(), trace, time(15.0)) is None


def test_mad_factor():
    trace = sta_lta_trace({(30.0, 31.0): 1.5})
    assert pick_one(StaLtaPicker(mad_factor=5.0), trace, time(30.0)) is not None
    assert pick_one(StaLtaPicker(mad_factor=20.0), trace, time(30.0)) is None


def test_closest_trigger():
    trace = sta_lta_trace({(28.0, 28.5): 3.0, (31.0, 31.5): 3.0})
    picker = StaLtaPicker()

    pick = pick_one(picker, trace, time(29.0))
    assert pick is not None
    assert pick.time == time(28.0)

    pick = pick_one(picker, trace, time(30.0))
    assert pick is not None
    assert pick.time == time(31.0)


def test_reject_pre_event_onsets():
    trace = sta_lta_trace({(28.0, 28.5): 3.0, (31.0, 31.5): 3.0})
    picker = StaLtaPicker()

    # The closest onset precedes the event, the later one is picked
    pick = pick_one(picker, trace, time(29.4), event_time=time(29.0))
    assert pick is not None
    assert pick.time == time(31.0)

    assert pick_one(picker, trace, time(31.2), event_time=time(32.0)) is None


def test_search_window():
    trace = sta_lta_trace({(30.0, 31.0): 3.0})
    assert pick_one(StaLtaPicker(), trace, time(33.0)) is None

    pick = pick_one(StaLtaPicker(search_window_seconds=8.0), trace, time(33.0))
    assert pick is not None
    assert pick.time == time(30.0)


def test_onset_before_search_window():
    """A trigger that is active at the window start has no onset in the window."""
    trace = sta_lta_trace({(29.0, 33.0): 3.0})
    assert pick_one(StaLtaPicker(), trace, time(32.0)) is None


def test_trigger_active_at_trace_start():
    trace = sta_lta_trace({(0.0, 2.0): 3.0})
    assert pick_one(StaLtaPicker(), trace, time(1.0)) is None


def test_trigger_active_at_trace_end():
    trace = sta_lta_trace({(59.0, 60.0): 3.0})
    pick = pick_one(StaLtaPicker(), trace, time(59.0))
    assert pick is not None
    assert pick.time == time(59.0)


def test_no_data():
    trace = sta_lta_trace({(30.0, 31.0): 3.0})
    assert pick_one(StaLtaPicker(), trace, time(120.0)) is None


@pytest.mark.parametrize(
    "position,min_delay,max_delay", [("centred", -0.1, 0.1), ("classic", 0.5, 2.0)]
)
def test_pick_characteristic_function(position, min_delay, max_delay):
    """Pick an impulsive arrival in white noise end-to-end."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(round(120 * SAMPLING_RATE))
    onset = round(60 * SAMPLING_RATE)
    data[onset : onset + 300] += (
        8 * rng.standard_normal(300) * np.exp(-np.arange(300) / 100)
    )
    trace = Trace("XX", "STA", "", "HHZ", TMIN, deltat=1 / SAMPLING_RATE, ydata=data)
    (char_function,) = to_pyrocko_traces(
        _compute_characteristic_functions(
            Stream([trace.to_obspy_trace()]), 2.0, 5.0, "energy", position
        )
    )
    picker = StaLtaPicker()

    pick = pick_one(picker, char_function, time(61.0))
    assert pick is not None
    delay = (pick.time - time(60.0)).total_seconds()
    assert min_delay <= delay <= max_delay
    assert pick.detection_value > 2.0

    assert pick_one(picker, char_function, time(30.0)) is None


@pytest.mark.asyncio
async def test_process_traces():
    rng = np.random.default_rng(0)
    traces = [
        Trace(
            "XX",
            "STA",
            "",
            channel,
            TMIN,
            deltat=1 / SAMPLING_RATE,
            ydata=rng.standard_normal(round(30 * SAMPLING_RATE)),
        )
        for channel in ("HHZ", "HHN", "HHE")
    ]
    images = await StaLta().process_traces(traces)

    assert [image.phase for image in images] == ["cake:S", "cake:P"]
    for image in images:
        assert image.image_function == "StaLta"
        assert image.n_traces == 1
    _, p_image = images
    assert p_image.traces[0].channel == "HHZ"


def test_log_onset():
    onset = np.array([0.001, 0.4, 1.0, np.e, 100.0])
    np.testing.assert_allclose(
        _log_onset(onset, min_onset_value=0.4),
        [np.log(0.4), np.log(0.4), 0.0, 1.0, np.log(100.0)],
    )


@pytest.mark.asyncio
async def test_process_traces_log_onset():
    """The image is the clipped log onset, noise is at 0 and arrivals are picked."""
    rng = np.random.default_rng(0)
    n_samples = round(60 * SAMPLING_RATE)
    onset = round(30 * SAMPLING_RATE)
    traces = []
    for channel in ("HHZ", "HHN", "HHE"):
        data = rng.standard_normal(n_samples)
        data[onset : onset + 300] += (
            20 * rng.standard_normal(300) * np.exp(-np.arange(300) / 100)
        )
        traces.append(
            Trace("XX", "STA", "", channel, TMIN, deltat=1 / SAMPLING_RATE, ydata=data)
        )

    # QuakeMigrate's default windows, a long STA window flattens the onset peak
    function = StaLta(sta_seconds=0.2, lta_seconds=1.0, min_onset_value=0.5)
    s_image, p_image = await function.process_traces(traces)

    for image in (p_image, s_image):
        (trace,) = image.traces
        assert trace.ydata.min() >= np.log(0.5)
        assert abs(np.median(trace.ydata)) < 0.1
        pick = function.picker.pick_trace(trace, image.phase, time(0), time(30.5))
        assert pick is not None
        assert abs((pick.time - time(30.0)).total_seconds()) <= 0.1
        assert pick.detection_value > np.log(10.0)


def test_picker_config():
    assert StaLta().position == "centred"
    with pytest.raises(ValidationError):
        StaLta(position="left")
    assert StaLta().min_onset_value == 0.4
    # QuakeMigrate's default windows
    assert (StaLta().sta_seconds, StaLta().lta_seconds) == (0.2, 1.0)
    assert StaLta().picker.mad_factor == 5.0
    with pytest.raises(ValidationError):
        StaLta(min_onset_value=0.001)

    function = StaLta.model_validate({"picker": {"mad_factor": 5.0}})
    assert function.picker.mad_factor == 5.0
    assert function.picker.search_window_seconds == 5.0

    loaded = StaLta.model_validate_json(function.model_dump_json())
    assert loaded.picker == function.picker

    for invalid in (
        {"mad_factor": 0.0},
        {"search_window_seconds": -1.0},
        {"unknown": 1.0},
    ):
        with pytest.raises(ValidationError):
            StaLtaPicker.model_validate(invalid)


@pytest.mark.parametrize("lta_seconds", [5.0, 30.0])
def test_blinding_covers_lta(lta_seconds):
    function = StaLta(sta_seconds=1.0, lta_seconds=lta_seconds)
    assert function.get_blinding().total_seconds() == pytest.approx(1.2 * lta_seconds)


def horizontal(channel: str, tmin_offset: float, n_samples: int, value: float):
    return Trace(
        "XX",
        "STA",
        "",
        channel,
        TMIN + tmin_offset,
        deltat=1 / SAMPLING_RATE,
        ydata=np.full(n_samples, value),
    )


def test_merge_horizontal_components():
    merged = _merge_horizontal_components(
        [horizontal("HHN", 0.0, 100, 3.0), horizontal("HHE", 0.0, 100, 4.0)]
    )
    assert len(merged) == 1
    np.testing.assert_allclose(merged[0].ydata, np.sqrt((9.0 + 16.0) / 2))


@pytest.mark.parametrize(
    "east",
    [
        # Gap in the east component
        [horizontal("HHE", 0.0, 40, 4.0), horizontal("HHE", 6.0, 40, 4.0)],
        # Same length, shifted start
        [horizontal("HHE", 0.5, 100, 4.0)],
    ],
)
def test_merge_misaligned_horizontal_components(east, caplog):
    north = horizontal("HHN", 0.0, 100, 3.0)
    merged = _merge_horizontal_components([north, *east])
    assert len(merged) == 1
    assert merged[0].ydata.size == 100
    assert "cannot merge misaligned" in caplog.text
