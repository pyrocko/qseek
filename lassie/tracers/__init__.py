from typing import Annotated, Union

from pydantic import BaseModel, Field

from lassie.tracers.cake import CakeTracer
from lassie.tracers.constant_velocity import ConstantVelocityTracer

Tracer = Annotated[
    Union[CakeTracer, ConstantVelocityTracer], Field(discriminator="tracer")
]


class Tracers(BaseModel):
    __root__: list[Tracer] = []
