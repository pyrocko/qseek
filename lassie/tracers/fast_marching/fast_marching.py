import asyncio
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel
from pyrocko.modelling import eikonal

from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    VelocityModels,
)

from ..base import ModelledArrival, RayTracer

if TYPE_CHECKING:
    from lassie.models.station import Stations
    from lassie.octree import Octree

logger = logging.getLogger(__name__)


class FastMarchingArrival(ModelledArrival):
    tracer: Literal["FastMarchingArrival"] = "FastMarchingArrival"
    phase: str


async def eikonal_fast_marching_solver(
    velocity_model: np.ndarray, times: np.ndarray, spacing: float
) -> np.ndarray:
    await asyncio.to_thread(
        eikonal.eikonal_solver_fmm_cartesian,
        velocity_model,
        times,
        delta=spacing,
    )
    return times


class FastMarchingPhaseRayTracer(BaseModel):
    phase: str

    velocity_model: VelocityModels = Constant3DVelocityModel(velocity=3000.0)

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        for station in stations:
            velocity_model = self.velocity_model.get_model(octree, station)

            logging.info(
                "forward modelling travel times for station %s...", station.pretty_nsl
            )
            arrival_times = await eikonal_fast_marching_solver(
                velocity_model.velocity_model,
                velocity_model.get_time_grid(),
                spacing=velocity_model.grid_spacing,
            )

            velocity_model.trim_to_octree(arrival_times, octree=octree)


class FastMarchingRayTracer(RayTracer):
    tracer: Literal["FastMarchingTracer"] = "FastMarchingTracer"

    tracers: dict[str, FastMarchingPhaseRayTracer] = {
        "P": FastMarchingPhaseRayTracer(phase="P"),
        "S": FastMarchingPhaseRayTracer(phase="S"),
    }

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        for tracer in self.tracers.values():
            await tracer.prepare(octree, stations)
