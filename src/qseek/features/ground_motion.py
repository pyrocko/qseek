from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from qseek.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from qseek.utils import ChannelSelectors

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection


class ReceiverGroundMotion(ReceiverFeature):
    feature: Literal["ReceiverGroundMotion"] = "ReceiverGroundMotion"

    seconds_before: float
    seconds_after: float
    peak_ground_acceleration: float
    peak_horizontal_acceleration: float
    peak_ground_velocity: float


class EventGroundMotion(EventFeature):
    feature: Literal["EventGroundMotion"] = "EventGroundMotion"

    seconds_before: float
    seconds_after: float
    peak_ground_acceleration: float
    peak_horizontal_acceleration: float
    peak_ground_velocity: float


def _get_maximum(traces: list[Trace]) -> float:
    data = np.array([tr.ydata for tr in traces])
    norm_traces = np.linalg.norm(data, axis=0)
    return float(norm_traces.max())


class GroundMotionExtractor(FeatureExtractor):
    feature: Literal["GroundMotion"] = "GroundMotion"

    seconds_before: float = 3.0
    seconds_after: float = 8.0

    async def add_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        receiver_motions: list[ReceiverGroundMotion] = []
        for receiver in event.receivers:
            try:
                traces_acc = receiver.get_waveforms_restituted(
                    squirrel,
                    seconds_after=self.seconds_after,
                    seconds_before=self.seconds_before,
                    quantity="acceleration",
                )
                traces_vel = receiver.get_waveforms_restituted(
                    squirrel,
                    seconds_after=self.seconds_after,
                    seconds_before=self.seconds_before,
                    quantity="velocity",
                )
                pga = _get_maximum(ChannelSelectors.All(traces_acc))
                pha = _get_maximum(ChannelSelectors.Horizontal(traces_acc))
                pgv = _get_maximum(ChannelSelectors.All(traces_vel))

                ground_motion = ReceiverGroundMotion(
                    seconds_before=self.seconds_before,
                    seconds_after=self.seconds_after,
                    peak_ground_acceleration=pga,
                    peak_horizontal_acceleration=pha,
                    peak_ground_velocity=pgv,
                )
            except Exception:
                continue
            receiver_motions.append(ground_motion)

        event_ground_motions = EventGroundMotion(
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            peak_ground_acceleration=max(
                gm.peak_ground_acceleration for gm in receiver_motions
            ),
            peak_horizontal_acceleration=max(
                gm.peak_ground_acceleration for gm in receiver_motions
            ),
            peak_ground_velocity=max(
                gm.peak_ground_velocity for gm in receiver_motions
            ),
        )
        event.add_feature(event_ground_motions)
