from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

# Has to be imported to register as subclass
from qseek.magnitudes import (
    local_magnitude,  # noqa: F401
    moment_magnitude,  # noqa: F401
)
from qseek.magnitudes.base import EventMagnitude, EventMagnitudeCalculator

type EventMagnitudeType = Annotated[
    Union[EventMagnitude.get_subclasses()],
    Field(discriminator="magnitude"),
]

type EventMagnitudeCalculatorType = Annotated[
    Union[EventMagnitudeCalculator.get_subclasses()],
    Field(discriminator="magnitude"),
]
