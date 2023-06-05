from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pyrocko.squirrel import Squirrel

from lassie.receiver_features.base import Feature, ReceiverFeatureExtractor

if TYPE_CHECKING:
    from lassie.models.detection import EventDetection


class MaximumAmplitudes(Feature):
    name: Literal["MaximumAmplitudes"] = "MaximumAmplitudes"
    counts_vertical: int
    counts_horizontal: int


class WaveformAmplitudes(ReceiverFeatureExtractor):
    name: Literal["WaveformAmplitudes"] = "WaveformAmplitudes"

    def get_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> list[MaximumAmplitudes]:
        detection = event.get_detection("P")
        for receiver in detection.receivers:
            receiver.get_waveforms(squirrel, seconds_after=10.0)
