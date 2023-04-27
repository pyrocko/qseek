from typing import Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.tracers.cake import CakeTracer
from lassie.tracers.constant_velocity import ConstantVelocityTracer

RayTracer = Annotated[
    Union[CakeTracer, ConstantVelocityTracer],
    Field(discriminator="tracer"),
]


class RayTracers(BaseModel):
    __root__: list[RayTracer] = []

    def __iter__(self) -> Iterator[RayTracer]:
        yield from self.__root__
