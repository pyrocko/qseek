from datetime import datetime

from pydantic import BaseModel


class PhaseArrival(BaseModel):
    time: datetime
