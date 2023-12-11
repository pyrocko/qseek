from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.features import (
    ground_motion,  # noqa: F401
    local_magnitude,  # noqa: F401
)
from qseek.features.base import EventFeature, FeatureExtractor, ReceiverFeature

FeatureExtractorType = Annotated[
    Union[FeatureExtractor.get_subclasses()],
    Field(..., discriminator="feature"),
]


ReceiverFeaturesTypes = Annotated[
    Union[ReceiverFeature.get_subclasses()],
    Field(..., discriminator="feature"),
]

EventFeaturesTypes = Annotated[
    Union[EventFeature.get_subclasses()],
    Field(..., discriminator="feature"),
]
