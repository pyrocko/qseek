from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from lassie.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from lassie.features.ground_motion import (
    EventGroundMotion,
    GroundMotionExtractor,
    ReceiverGroundMotion,
)
from lassie.features.local_magnitude import (
    LocalMagnitude,
    LocalMagnitudeExtractor,
    StationMagnitude,
)

FeatureExtractors = Annotated[
    Union[GroundMotionExtractor, LocalMagnitudeExtractor, FeatureExtractor],
    Field(..., discriminator="feature"),
]


ReceiverFeaturesTypes = Annotated[
    Union[ReceiverGroundMotion, StationMagnitude, ReceiverFeature],
    Field(..., discriminator="feature"),
]

EventFeaturesTypes = Annotated[
    Union[EventGroundMotion, LocalMagnitude, EventFeature],
    Field(..., discriminator="feature"),
]
