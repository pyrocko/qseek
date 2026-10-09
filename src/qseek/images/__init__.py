from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

from qseek.images.base import ImageFunction
from qseek.images.seisbench import SeisBench
from qseek.images.sta_lta import StaLta

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type ImageFunctionType = ImageFunction
else:
    type ImageFunctionType = Annotated[
        Union[ImageFunction.get_subclasses()],
        Field(discriminator="image"),
    ]

__all__ = [
    "ImageFunction",
    "ImageFunctionType",
    "SeisBench",
    "StaLta",
]
