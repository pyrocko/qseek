from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.waveforms.base import WaveformProvider
from qseek.waveforms.sds import SDSArchive  # noqa: F401
from qseek.waveforms.seedlink import SeedLink  # noqa: F401
from qseek.waveforms.squirrel import PyrockoSquirrel  # noqa: F401

type WaveformProviderType = Annotated[
    Union[WaveformProvider.get_subclasses()],
    Field(discriminator="provider"),
]
