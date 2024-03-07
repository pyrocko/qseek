from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import Field, RootModel

from qseek.tracers import (
    cake,  # noqa: F401
    constant_velocity,  # noqa: F401
)
from qseek.tracers.base import RayTracer

if TYPE_CHECKING:
    from qseek.models.station import Stations
    from qseek.octree import Octree
    from qseek.utils import PhaseDescription

logger = logging.getLogger(__name__)

RayTracerType = Annotated[
    Union[(RayTracer, *RayTracer.get_subclasses())],
    Field(..., discriminator="tracer"),
]


class RayTracers(RootModel):
    root: list[RayTracerType] = []

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        phases: tuple[PhaseDescription, ...],
        rundir: Path | None = None,
    ) -> None:
        prepared_tracers = []
        for phase in phases:
            tracer = self.get_phase_tracer(phase)
            if tracer in prepared_tracers:
                continue
            phases = tracer.get_available_phases()
            logger.info(
                "preparing ray tracer %s for phase %s", tracer.tracer, ", ".join(phases)
            )
            await tracer.prepare(octree, stations, rundir)
            prepared_tracers.append(tracer)

    def get_available_phases(self) -> tuple[str, ...]:
        phases = []
        for tracer in self:
            phases.extend([*tracer.get_available_phases()])
        if len(set(phases)) != len(phases):
            duplicate_phases = {phase for phase in phases if phases.count(phase) > 1}
            raise ValueError(
                f"Phases {', '.join(duplicate_phases)} was provided twice."
                " Rename or remove the duplicate phases from the tracers."
            )
        return tuple(phases)

    def get_phase_tracer(self, phase: str) -> RayTracer:
        for tracer in self:
            if phase in tracer.get_available_phases():
                return tracer
        raise ValueError(
            f"No tracer found for phase {phase}."
            " Please add a tracer for this phase or rename the phase to match a tracer."
            f" Available phases: {', '.join(self.get_available_phases())}."
        )

    def __iter__(self) -> Iterator[RayTracer]:
        yield from self.root

    def iter_phase_tracer(
        self, phases: tuple[PhaseDescription, ...]
    ) -> Iterator[tuple[PhaseDescription, RayTracer]]:
        for phase in phases:
            tracer = self.get_phase_tracer(phase)
            yield (phase, tracer)
