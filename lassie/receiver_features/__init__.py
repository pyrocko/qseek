from typing import Annotated, Union

from pydantic import Field

from lassie.receiver_features.amplitudes import WaveformAmplitudes
from lassie.receiver_features.base import ReceiverFeatureExtractor

FeatureExtraction = Annotated[
    Union[ReceiverFeatureExtractor, WaveformAmplitudes], Field(descriminator="name")
]
