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

type FeatureExtractorType = Annotated[
    Union[FeatureExtractor.get_subclasses()],
    Field(discriminator="feature"),
]

type ReceiverFeaturesType = Annotated[
    Union[ReceiverFeature.get_subclasses()],
    Field(discriminator="feature"),
]

type EventFeaturesType = Annotated[
    Union[EventFeature.get_subclasses()],
    Field(discriminator="feature"),
]
