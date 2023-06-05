from typing import Annotated, Union

from pydantic import Field

from lassie.features_event.base import EventFeatureExtractor
from lassie.features_event.magnitude_local import WaveformAmplitudes

EventFeatures = Annotated[
    Union[WaveformAmplitudes, EventFeatureExtractor], Field(discriminator="feature")
]
