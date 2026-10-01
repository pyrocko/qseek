from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.images.base import ImageFunction
from qseek.images.seisbench import SeisBench
from qseek.images.sta_lta import StaLta

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
