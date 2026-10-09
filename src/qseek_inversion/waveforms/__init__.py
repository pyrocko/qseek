from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

from qseek_inversion.waveforms.base import WaveformSelection
from qseek_inversion.waveforms.event_selection import (
    EventWaveformsSelection,  # noqa: F401
)
from qseek_inversion.waveforms.synthetics import SyntheticEvents  # noqa: F401

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    WaveformSelectionType = WaveformSelection
else:
    WaveformSelectionType = Annotated[
        Union[(WaveformSelection, *WaveformSelection.get_subclasses())],
        Field(..., discriminator="waveforms"),
    ]
