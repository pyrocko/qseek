from typing import Annotated, Union

from pydantic import Field

from lassie.features_receiver.base import ReceiverFeatureExtractor
from lassie.features_receiver.waveform_amplitudes import WaveformAmplitudes

ReceiverFeatures = Annotated[
    Union[WaveformAmplitudes, ReceiverFeatureExtractor], Field(discriminator="feature")
]
