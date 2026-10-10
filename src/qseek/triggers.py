"""Triggers turn the detection function of a search into detections."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Literal, Union

import numpy as np
from pydantic import Field, PositiveFloat
from scipy import signal, stats

from qseek.base import Model
from qseek.utils import RangeType

logger = logging.getLogger(__name__)

THRESHOLD_RANGE_DESCRIPTION = (
    "Range of the threshold as `[min, max]` semblance. The adaptive thresholds, height"
    " and prominence, are clipped to it: the minimum keeps noise peaks out of quiet or"
    " flat windows, the maximum keeps busy windows, e.g. during a swarm, from raising"
    " their own threshold. If `null`, the thresholds are not clipped."
)


def _clip(
    height: float, prominence: float, threshold_range: RangeType | None
) -> tuple[float, float]:
    if threshold_range is None:
        return height, prominence
    low, high = threshold_range
    return min(max(height, low), high), min(max(prominence, low), high)


class Trigger(Model):
    """Base class of the triggers.

    A trigger detects events as peaks of the detection function, the maximum
    semblance over all nodes. A peak is a detection when its height and its
    prominence exceed the thresholds of the trigger. Subclasses set the thresholds
    with `get_threshold`.
    """

    trigger: Literal["Trigger"] = "Trigger"

    blinding: timedelta = Field(
        default=timedelta(seconds=1.0),
        ge=timedelta(0),
        description="Minimum time between two detections. Peaks of the detection"
        " function closer than this count as one detection. Prevents detecting the"
        " same event twice.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[Trigger], ...]:
        """Get the subclasses of this class.

        Returns:
            tuple[type[Trigger], ...]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    def get_threshold(self, detection_function: np.ndarray) -> tuple[float, float]:
        """Get the thresholds for a window of the detection function.

        Args:
            detection_function (np.ndarray): Detection function of the window,
                without padding.

        Returns:
            tuple[float, float]: Minimum height and minimum prominence of a
                detection.
        """
        raise NotImplementedError

    async def detect(
        self,
        detection_function: np.ndarray,
        sampling_rate: float,
        padding: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect events in a window of the detection function.

        Peaks in the padding are discarded. The padding gives the peaks at the
        edges of the window their prominence.

        Args:
            detection_function (np.ndarray): Detection function of the window,
                with padding.
            sampling_rate (float): Sampling rate of the detection function in Hz.
            padding (int, optional): Number of padding samples at both ends.
                Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sample indices of the detections in the
                window without padding, and the detection function at the
                detections.
        """
        if padding < 0:
            raise ValueError("padding must not be negative")

        window = detection_function[padding : detection_function.size - padding]
        height, prominence = self.get_threshold(window)
        logger.debug("trigger height %g, prominence %g", height, prominence)

        peak_idx, _ = await asyncio.to_thread(
            signal.find_peaks,
            detection_function,
            height=float(height),
            prominence=float(prominence),
            distance=max(1, round(self.blinding.total_seconds() * sampling_rate)),
        )
        peak_idx -= padding
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < window.size)]
        return peak_idx, window[peak_idx]


class MADTrigger(Trigger):
    """Threshold that adapts to the noise level of each window."""

    trigger: Literal["MADTrigger"] = "MADTrigger"

    mad_factor: PositiveFloat = Field(
        default=10.0,
        description="The threshold is this factor times the median absolute deviation"
        " (MAD) of the detection function in each processed window.",
    )
    threshold_range: RangeType | None = Field(
        default=None,
        description=THRESHOLD_RANGE_DESCRIPTION,
    )

    def get_threshold(self, detection_function: np.ndarray) -> tuple[float, float]:
        """Get the thresholds from the MAD of the detection function.

        Args:
            detection_function (np.ndarray): Detection function of the window,
                without padding.

        Returns:
            tuple[float, float]: Minimum height and minimum prominence of a
                detection, both the MAD times `mad_factor`, clipped to the
                `threshold_range`.
        """
        threshold = stats.median_abs_deviation(detection_function) * self.mad_factor
        return _clip(threshold, threshold, self.threshold_range)


class ModZScoreTrigger(Trigger):
    """Threshold on the modified z-score of the detection function in each window."""

    trigger: Literal["ModZScoreTrigger"] = "ModZScoreTrigger"

    z_score: PositiveFloat = Field(
        default=7.0,
        description="Minimum modified z-score of a detection. The modified z-score is"
        " the distance from the median of the detection function in each processed"
        " window, in units of the MAD scaled to a standard deviation (MAD / 0.6745).",
    )
    threshold_range: RangeType | None = Field(
        default=None,
        description=THRESHOLD_RANGE_DESCRIPTION,
    )

    def get_threshold(self, detection_function: np.ndarray) -> tuple[float, float]:
        """Get the thresholds from the median and the MAD of the detection function.

        Args:
            detection_function (np.ndarray): Detection function of the window,
                without padding.

        Returns:
            tuple[float, float]: Minimum height and minimum prominence of a
                detection. The height is the median plus `z_score` scaled MADs, the
                prominence `z_score` scaled MADs, both clipped to the
                `threshold_range`.
        """
        sigma = stats.median_abs_deviation(detection_function, scale="normal")
        prominence = self.z_score * sigma
        height = float(np.median(detection_function)) + prominence
        return _clip(height, prominence, self.threshold_range)


class ThresholdTrigger(Trigger):
    """Fixed semblance threshold for all windows."""

    trigger: Literal["ThresholdTrigger"] = "ThresholdTrigger"

    threshold: PositiveFloat = Field(
        default=0.3,
        description="Minimum semblance of a detection.",
    )

    def get_threshold(self, detection_function: np.ndarray) -> tuple[float, float]:
        """Get the fixed thresholds.

        Args:
            detection_function (np.ndarray): Detection function of the window,
                without padding.

        Returns:
            tuple[float, float]: Minimum height and minimum prominence of a
                detection, both `threshold`.
        """
        return self.threshold, self.threshold


# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type TriggerType = Trigger
else:
    type TriggerType = Annotated[
        Union[Trigger.get_subclasses()],
        Field(discriminator="trigger"),
    ]
