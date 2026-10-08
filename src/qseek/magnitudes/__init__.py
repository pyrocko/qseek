from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

# Has to be imported to register as subclass
from qseek.magnitudes import (
    local_magnitude,  # noqa: F401
    moment_magnitude,  # noqa: F401
)
from qseek.magnitudes.base import EventMagnitude, EventMagnitudeCalculator

# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type EventMagnitudeType = EventMagnitude
else:
    type EventMagnitudeType = Annotated[
        Union[EventMagnitude.get_subclasses()],
        Field(discriminator="magnitude"),
    ]

if TYPE_CHECKING:
    type EventMagnitudeCalculatorType = EventMagnitudeCalculator
else:
    type EventMagnitudeCalculatorType = Annotated[
        Union[EventMagnitudeCalculator.get_subclasses()],
        Field(discriminator="magnitude"),
    ]
