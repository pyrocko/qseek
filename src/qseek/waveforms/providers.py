from typing import Annotated, Union

from pydantic import Field

from qseek.waveforms.base import WaveformProvider
from qseek.waveforms.seedlink import SeedLink  # noqa: F401
from qseek.waveforms.squirrel import PyrockoSquirrel  # noqa: F401

WaveformProviderType = Annotated[
    Union[(WaveformProvider, *WaveformProvider.get_subclasses())],
    Field(..., discriminator="provider"),
]
