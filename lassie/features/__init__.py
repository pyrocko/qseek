from typing import Annotated, Union

from pydantic import Field

from lassie.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from lassie.features.ground_motion import (
    EventGroundMotion,
    GroundMotionExtractor,
    ReceiverGroundMotion,
)
from lassie.features.local_magnitude import (
    LocalMagnitudeExtractor,
    ReceiverLocalMagnitude,
)

FeatureExtractors = Annotated[
    Union[GroundMotionExtractor, LocalMagnitudeExtractor, FeatureExtractor],
    Field(discriminator="feature"),
]


ReceiverFeatures = Annotated[
    Union[ReceiverGroundMotion, ReceiverLocalMagnitude, ReceiverFeature],
    Field(discriminator="feature"),
]

EventFeatures = Annotated[
    Union[EventGroundMotion, EventFeature],
    Field(discriminator="feature"),
]
