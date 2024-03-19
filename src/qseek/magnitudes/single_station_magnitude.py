from typing import TYPE_CHECKING, Literal

from pydantic import Field, PositiveFloat

from qseek.magnitudes.base import EventMagnitude, EventMagnitudeCalculator
from qseek.utils import NSL

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection


class SingleStationMagnitude(EventMagnitude):
    magnitude: Literal["SingleStationMagnitude"] = "SingleStationMagnitude"

    error: float
    average: float


class SingleStationMagnitudeCalculator(EventMagnitudeCalculator):
    magnitude: Literal["SingleStationMagnitude"] = "SingleStationMagnitude"

    stations: NSL = Field(
        default=None,
        description="The stations to use for calculating the single station magnitude.",
    )
    seconds_before: PositiveFloat = Field(
        default=10.0,
        description="Seconds before first phase arrival to extract.",
    )
    seconds_after: PositiveFloat = Field(
        default=10.0,
        description="Seconds after last phase arrival to extract.",
    )
    padding_seconds: PositiveFloat = Field(
        default=10.0,
        description="Seconds padding before and after the extraction window.",
    )
    factors: tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="Factors to multiply the amplitude by to get the magnitude.",
    )

    async def add_magnitude(
        self, squirrel: Squirrel, event: EventDetection
    ) -> None: ...
