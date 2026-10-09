from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

from qseek.waveforms.base import WaveformProvider
from qseek.waveforms.sds import SDSArchive  # noqa: F401
from qseek.waveforms.seedlink import SeedLink  # noqa: F401
from qseek.waveforms.squirrel import PyrockoSquirrel  # noqa: F401

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type WaveformProviderType = WaveformProvider
else:
    type WaveformProviderType = Annotated[
        Union[WaveformProvider.get_subclasses()],
        Field(discriminator="provider"),
    ]
