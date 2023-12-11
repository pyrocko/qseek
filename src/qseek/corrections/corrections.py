from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.corrections.base import StationCorrections

StationCorrectionType = Annotated[
    Union[StationCorrections.get_subclasses()],
    Field(..., discriminator="corrections"),
]
