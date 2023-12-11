from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.features import ground_motion, local_magnitude  # noqa: F401
from qseek.features.base import EventFeature, FeatureExtractor, ReceiverFeature

FeatureExtractorType = Annotated[
    Union[(FeatureExtractor, *FeatureExtractor.get_subclasses())],
    Field(..., discriminator="feature"),
]


ReceiverFeaturesTypes = Annotated[
    Union[(ReceiverFeature, *ReceiverFeature.get_subclasses())],
    Field(..., discriminator="feature"),
]

EventFeaturesTypes = Annotated[
    Union[(EventFeature, *EventFeature.get_subclasses())],
    Field(..., discriminator="feature"),
]
