from typing import Annotated, Union

from pydantic import Field

# Has to be imported to register as subclass
from qseek.magnitudes import (
    local_magnitude,  # noqa: F401
    moment_magnitude,  # noqa: F401
)
from qseek.magnitudes.base import EventMagnitude, EventMagnitudeCalculator

EventMagnitudeType = Annotated[
    Union[(EventMagnitude, *EventMagnitude.get_subclasses())],
    Field(..., discriminator="magnitude"),
]

EventMagnitudeCalculatorType = Annotated[
    Union[(EventMagnitudeCalculator, *EventMagnitudeCalculator.get_subclasses())],
    Field(..., discriminator="magnitude"),
]
