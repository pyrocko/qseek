from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PhaseArrival(BaseModel):
    name: Literal["PhaseArrival"] = "PhaseArrival"
    time: datetime
