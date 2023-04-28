from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.tracers.cake import CakeTracer
from lassie.tracers.constant_velocity import ConstantVelocityTracer

if TYPE_CHECKING:
    from lassie.models.receiver import Receivers
    from lassie.octree import Octree
    from lassie.tracers.base import RayTracer

RayTracerType = Annotated[
    Union[CakeTracer, ConstantVelocityTracer],
    Field(discriminator="tracer"),
]


class RayTracers(BaseModel):
    __root__: list[RayTracerType] = Field([], alias="tracers")

    def set_octree(self, octree: Octree) -> None:
        for tracer in self:
            tracer.set_octree(octree)

    def set_receivers(self, receivers: Receivers) -> None:
        for tracer in self:
            tracer.set_receivers(receivers)

    def __iter__(self) -> Iterator[RayTracer]:
        yield from self.__root__
