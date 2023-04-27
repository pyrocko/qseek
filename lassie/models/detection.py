from datetime import datetime
from uuid import UUID, uuid4

from pydantic import Field

from lassie.models.location import Location


class Detection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    detection_peak: float
