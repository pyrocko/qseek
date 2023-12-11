from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.corrections.base import StationCorrections
from qseek.corrections.simple import SimpleCorrections  # noqa: F401

StationCorrectionType = Annotated[
    Union[StationCorrections.get_subclasses()],
    Field(..., discriminator="corrections"),
]
