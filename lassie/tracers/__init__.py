from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import BaseModel, Field

from lassie.tracers.cake import CakeTracer
from lassie.tracers.constant_velocity import ConstantVelocityTracer
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from lassie.models.station import Stations
    from lassie.octree import Octree
    from lassie.tracers.base import RayTracer

RayTracerType = Annotated[
    Union[CakeTracer, ConstantVelocityTracer],
    Field(discriminator="tracer"),
]


class RayTracers(BaseModel):
    __root__: list[RayTracerType] = []

    def set_octree(self, octree: Octree) -> None:
        for tracer in self:
            tracer.set_octree(octree)

    def set_receivers(self, stations: Stations) -> None:
        for tracer in self:
            tracer.set_stations(stations)

    def get_available_phases(self) -> tuple[str]:
        phases = []
        for tracer in self:
            phases.extend([*tracer.get_available_phases()])
        if len(set(phases)) != len(phases):
            raise ValueError("A phase provided twice")
        return tuple(phases)

    def get_phase_tracer(self, phase: str) -> RayTracer:
        for tracer in self:
            if phase in tracer.get_available_phases():
                return tracer
        raise ValueError(
            f"No tracer found for phase {phase}."
            f" Available phases: {', '.join(self.get_available_phases())}"
        )

    def __iter__(self) -> Iterator[RayTracer]:
        yield from self.__root__

    def iter_phase_tracer(self) -> Iterator[tuple[PhaseDescription, RayTracer]]:
        for tracer in self:
            for phase in tracer.get_available_phases():
                yield (phase, tracer)
