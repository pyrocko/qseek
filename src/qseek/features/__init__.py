from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

# Has to be imported to register as subclass
from qseek.features import ground_motion  # noqa: F401
from qseek.features.base import (
    EventFeature,
    FeatureExtractor,
    ReceiverFeature,
)

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type FeatureExtractorType = FeatureExtractor
else:
    type FeatureExtractorType = Annotated[
        Union[FeatureExtractor.get_subclasses()],
        Field(discriminator="feature"),
    ]

if TYPE_CHECKING:
    type ReceiverFeaturesType = ReceiverFeature
else:
    type ReceiverFeaturesType = Annotated[
        Union[ReceiverFeature.get_subclasses()],
        Field(discriminator="feature"),
    ]

if TYPE_CHECKING:
    type EventFeaturesType = EventFeature
else:
    type EventFeaturesType = Annotated[
        Union[EventFeature.get_subclasses()],
        Field(discriminator="feature"),
    ]
