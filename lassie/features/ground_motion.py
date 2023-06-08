from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from lassie.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.models.detection import EventDetection


class ReceiverGroundMotion(ReceiverFeature):
    feature: Literal["ReceiverGroundMotion"] = "ReceiverGroundMotion"

    phase: PhaseDescription
    seconds_before: float
    seconds_after: float
    peak_ground_acceleration: float
    peak_horizontal_acceleration: float
    peak_ground_velocity: float


class EventGroundMotion(EventFeature):
    feature: Literal["EventGroundMotion"] = "EventGroundMotion"
    phase: PhaseDescription
    seconds_before: float
    seconds_after: float
    peak_ground_acceleration: float
    peak_horizontal_acceleration: float
    peak_ground_velocity: float


@dataclass
class ChannelSelector:
    channels: str
    number_channels: int


class ChannelSelectors:
    Horizontal = ChannelSelector("EN23", 2)
    Vertical = ChannelSelector("Z0", 1)
    All = ChannelSelector("ENZ0123", 3)


def _get_maximum(traces: list[Trace], selector: ChannelSelector) -> float:
    traces = [tr for tr in traces if tr.channel[-1] in selector.channels]
    if len(traces) != selector.number_channels:
        raise KeyError(
            "cannot get %d channels for selector %s",
            selector.number_channels,
            selector.channels,
        )

    data = np.array([tr.ydata for tr in traces])
    norm_traces = np.linalg.norm(data, axis=0)
    return float(norm_traces.max())


class GroundMotionExtractor(FeatureExtractor):
    feature: Literal["GroundMotion"] = "GroundMotion"

    phase: PhaseDescription = "cake:S"
    seconds_before: float = 3.0
    seconds_after: float = 8.0

    async def add_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        receiver_motions: list[ReceiverGroundMotion] = []
        for receiver in event.receivers:
            if self.phase not in receiver.phase_arrivals:
                continue

            traces_acc = receiver.get_waveforms_restituted(
                squirrel,
                phase=self.phase,
                seconds_after=self.seconds_after,
                seconds_before=self.seconds_before,
                quantity="acceleration",
            )
            traces_vel = receiver.get_waveforms_restituted(
                squirrel,
                phase=self.phase,
                seconds_after=self.seconds_after,
                seconds_before=self.seconds_before,
                quantity="velocity",
            )
            try:
                pga = _get_maximum(traces_acc, ChannelSelectors.All)
                pha = _get_maximum(traces_acc, ChannelSelectors.Horizontal)
                pgv = _get_maximum(traces_vel, ChannelSelectors.All)

                ground_motion = ReceiverGroundMotion(
                    phase=self.phase,
                    seconds_before=self.seconds_before,
                    seconds_after=self.seconds_after,
                    peak_ground_acceleration=pga,
                    peak_horizontal_acceleration=pha,
                    peak_ground_velocity=pgv,
                )
            except KeyError:
                continue
            receiver_motions.append(ground_motion)
            receiver.add_feature(ground_motion)

        event_ground_motions = EventGroundMotion(
            phase=self.phase,
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
