from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

# Has to be imported to register as subclass
from qseek.features import ground_motion  # noqa: F401
from qseek.features.base import (
    EventFeature,
    FeatureExtractor,
    ReceiverFeature,
)

FeatureExtractorType = Annotated[
    Union[(FeatureExtractor, *FeatureExtractor.get_subclasses())],
    Field(..., discriminator="feature"),
]

ReceiverFeaturesType = Annotated[
    Union[(ReceiverFeature, *ReceiverFeature.get_subclasses())],
    Field(..., discriminator="feature"),
]

EventFeaturesType = Annotated[
    Union[(EventFeature, *EventFeature.get_subclasses())],
    Field(..., discriminator="feature"),
]
