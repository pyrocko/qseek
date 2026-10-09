from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field

from qseek.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from qseek.utils import ChannelSelectors

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection
    from qseek.models.station import StationInventory
    from qseek.waveforms.base import WaveformProvider


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
    """Peak ground motions of the detected events.

    The peak ground acceleration (PGA), peak horizontal acceleration and peak
    ground velocity (PGV) are measured on the restituted waveforms of each
    receiver. The event features hold the maxima over all receivers.
    """

    feature: Literal["GroundMotion"] = "GroundMotion"

    seconds_before: float = Field(
        default=3.0,
        description="Start of the measurement window in seconds before the first "
        "arrival at the receiver.",
    )
    seconds_after: float = Field(
        default=8.0,
        description="End of the measurement window in seconds after the last "
        "arrival at the receiver.",
    )

    async def add_features(
        self,
        waveform_provider: WaveformProvider,
        stations: StationInventory,
        event: EventDetection,
    ) -> None:
        receiver_motions: list[ReceiverGroundMotion] = []
        for receiver in event.receivers:
            try:
                traces_acc = await event.receivers.get_waveforms_restituted(
                    waveform_provider,
                    stations,
                    receivers=[receiver],
                    seconds_after=self.seconds_after,
                    seconds_before=self.seconds_before,
                    quantity="acceleration",
                )
                traces_vel = await event.receivers.get_waveforms_restituted(
                    waveform_provider,
                    stations,
                    receivers=[receiver],
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

        if not receiver_motions:
            return

        event_ground_motions = EventGroundMotion(
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            peak_ground_acceleration=max(
                gm.peak_ground_acceleration for gm in receiver_motions
            ),
            peak_horizontal_acceleration=max(
                gm.peak_horizontal_acceleration for gm in receiver_motions
            ),
            peak_ground_velocity=max(
                gm.peak_ground_velocity for gm in receiver_motions
            ),
        )
        event.add_feature(event_ground_motions)
