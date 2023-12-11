from typing import Annotated, Union

from pydantic import Field

from qseek.waveforms.base import WaveformProvider
from qseek.waveforms.squirrel import PyrockoSquirrel

WaveformProviderType = Annotated[
    Union[PyrockoSquirrel, WaveformProvider],
    Field(..., discriminator="provider"),
]
