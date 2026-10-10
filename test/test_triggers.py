from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from qseek.search import Search
from qseek.triggers import MADTrigger, ModZScoreTrigger, ThresholdTrigger

SAMPLING_RATE = 100.0


def detection_function(peaks: dict[int, float], n_samples: int = 2000) -> np.ndarray:
    data = np.zeros(n_samples, dtype=np.float32)
    for idx, height in peaks.items():
        data[idx] = height
    return data


@pytest.mark.asyncio
async def test_threshold_trigger() -> None:
    data = detection_function({300: 0.5, 900: 0.1, 1500: 0.8})
    trigger = ThresholdTrigger(threshold=0.2)

    idx, values = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
    np.testing.assert_array_equal(idx, [300, 1500])
    np.testing.assert_allclose(values, [0.5, 0.8])


@pytest.mark.asyncio
async def test_trigger_blinding() -> None:
    # Two peaks 0.3 s apart
    data = detection_function({1000: 0.5, 1030: 0.8})

    trigger = ThresholdTrigger(threshold=0.2, blinding=timedelta(seconds=0.5))
    idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
    np.testing.assert_array_equal(idx, [1030])

    trigger = ThresholdTrigger(threshold=0.2, blinding=timedelta(seconds=0.2))
    idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
    np.testing.assert_array_equal(idx, [1000, 1030])

    trigger = ThresholdTrigger(threshold=0.2, blinding=timedelta(0))
    idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
    np.testing.assert_array_equal(idx, [1000, 1030])


@pytest.mark.asyncio
async def test_trigger_padding() -> None:
    padding = 200
    data = detection_function({100: 0.9, 250: 0.5, 1700: 0.6, 1900: 0.9})
    trigger = ThresholdTrigger(threshold=0.2)

    idx, values = await trigger.detect(
        data, sampling_rate=SAMPLING_RATE, padding=padding
    )
    # Peaks in the padding are dropped, indices refer to the unpadded window
    np.testing.assert_array_equal(idx, [50, 1500])
    np.testing.assert_allclose(values, [0.5, 0.6])

    # A peak at the last sample of the window keeps its prominence from the padding
    data = detection_function({1799: 0.5}) + 0.25
    idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE, padding=padding)
    np.testing.assert_array_equal(idx, [1599])

    with pytest.raises(ValueError, match="padding"):
        await trigger.detect(data, sampling_rate=SAMPLING_RATE, padding=-1)


@pytest.mark.asyncio
async def test_mad_trigger() -> None:
    rng = np.random.default_rng(42)
    padding = 1000
    data = rng.uniform(0.0, 0.05, size=3000).astype(np.float32)
    # Noise in the padding does not raise the threshold
    data[:padding] += rng.uniform(0.0, 1.0, size=padding).astype(np.float32)
    data[-padding:] += rng.uniform(0.0, 1.0, size=padding).astype(np.float32)
    data[1500] = 3.0
    data[1800] = 0.4

    trigger = MADTrigger(mad_factor=10.0)
    window = data[padding:-padding]
    threshold = stats.median_abs_deviation(window) * 10.0
    assert trigger.get_threshold(window) == pytest.approx((threshold, threshold))
    threshold_padded, _ = trigger.get_threshold(data)
    assert threshold < 0.4 < threshold_padded

    idx, values = await trigger.detect(
        data, sampling_rate=SAMPLING_RATE, padding=padding
    )
    np.testing.assert_array_equal(idx, [500, 800])
    np.testing.assert_allclose(values, [3.0, 0.4])


@pytest.mark.asyncio
async def test_mod_z_score_trigger() -> None:
    rng = np.random.default_rng(42)
    padding = 200
    # Noise floor at 0.5 semblance
    data = rng.normal(0.5, 0.01, size=3000).astype(np.float32)
    data[1000] = 0.65
    data[2000] = 0.56

    trigger = ModZScoreTrigger(z_score=7.0)
    window = data[padding:-padding]
    sigma = stats.median_abs_deviation(window, scale="normal")
    assert sigma == pytest.approx(0.01, rel=0.1)
    height, prominence = trigger.get_threshold(window)
    assert prominence == pytest.approx(7.0 * sigma)
    assert height == pytest.approx(np.median(window) + 7.0 * sigma)

    idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE, padding=padding)
    np.testing.assert_array_equal(idx, [1000 - padding])

    # The height threshold of the MAD trigger ignores the noise floor
    mad_height, _ = MADTrigger(mad_factor=10.0).get_threshold(window)
    assert mad_height < np.median(window)


@pytest.mark.asyncio
async def test_threshold_range() -> None:
    # Flat window with small wiggles: the MAD is 0, every local maximum triggers
    data = detection_function({500: 0.2})
    data[100::50] += 0.001
    idx, _ = await MADTrigger().detect(data, sampling_rate=SAMPLING_RATE)
    assert idx.size > 1

    for trigger in (
        MADTrigger(threshold_range=(0.05, 1.0)),
        ModZScoreTrigger(threshold_range=(0.05, 1.0)),
    ):
        assert trigger.get_threshold(data) == pytest.approx((0.05, 0.05))
        idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
        np.testing.assert_array_equal(idx, [500])

    # Busy window: the events raise the MAD above the weaker event
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 0.5, size=2000).astype(np.float32)
    data[1000] = 1.0
    threshold, _ = MADTrigger().get_threshold(data)
    assert threshold > 1.0

    for trigger in (
        MADTrigger(threshold_range=(0.0, 0.2)),
        ModZScoreTrigger(threshold_range=(0.0, 0.2)),
    ):
        height, prominence = trigger.get_threshold(data)
        assert height <= 0.2
        assert prominence <= 0.2
        idx, _ = await trigger.detect(data, sampling_rate=SAMPLING_RATE)
        assert 1000 in idx

    with pytest.raises(ValidationError):
        MADTrigger.model_validate({"threshold_range": [0.2, 0.1]})


def test_trigger_blinding_not_negative() -> None:
    with pytest.raises(ValidationError):
        MADTrigger(blinding=timedelta(seconds=-1))


def test_search_trigger() -> None:
    assert Search().trigger == ThresholdTrigger(threshold=0.3)

    search = Search.model_validate(
        {"trigger": {"trigger": "ThresholdTrigger", "threshold": 0.5}}
    )
    assert search.trigger == ThresholdTrigger(threshold=0.5)
    search = Search.model_validate({"trigger": search.trigger.model_dump(mode="json")})
    assert search.trigger == ThresholdTrigger(threshold=0.5)


@pytest.mark.parametrize("field", ["detection_threshold", "detection_blinding"])
def test_search_detection_threshold_removed(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Search.model_validate({field: "MAD" if field == "detection_threshold" else 1})
