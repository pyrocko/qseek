from datetime import datetime
from typing import Literal

from pyrocko.squirrel import Squirrel

from lassie.features.base import FeatureExtractor, ReceiverFeature
from lassie.models.detection import EventDetection
from lassie.utils import PhaseDescription


class ReceiverLocalMagnitude(ReceiverFeature):
    feature: Literal["ReceiverLocalMagnitude"] = "ReceiverLocalMagnitude"

    local_magnitude: float
    peak_velocity: float
    peak_velocity_time: datetime


class LocalMagnitudeExtractor(FeatureExtractor):
    feature: Literal["LocalMagnitude"] = "LocalMagnitude"
    phase: PhaseDescription = "cake:S"

    async def add_features(self, squirrel: Squirrel, event: EventDetection) -> None:
        for receiver in event.receivers:
            receiver.get_waveforms_restituted(
                squirrel,
                phase=self.phase,
                seconds_before=3.0,
                seconds_after=10.0,
                quantity="velocity",
            )
