from typing import Annotated, Union

from pydantic import Field

from lassie.waveforms.base import WaveformProvider
from lassie.waveforms.squirrel import PyrockoSquirrel

WaveformProviderType = Annotated[
    Union[PyrockoSquirrel, WaveformProvider],
    Field(..., discriminator="provider"),
]
