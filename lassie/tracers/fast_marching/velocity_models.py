import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from lassie.models.location import Location

if TYPE_CHECKING:
    from lassie.models.station import Station
    from lassie.octree import Octree


logger = logging.getLogger(__name__)


class StationVelocityArrivalGrid3D(BaseModel):
    origin: Location
    station: Station

    grid_spacing: float

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]

    # Replace with extent and calculate on the fly
    north_coords: np.ndarray = PrivateAttr(None)
    east_coords: np.ndarray = PrivateAttr(None)
    depth_coords: np.ndarray = PrivateAttr(None)

    velocity_model: np.ndarray = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing
        self.east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self.north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self.depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1] + grid_spacing,
            grid_spacing,
        )

        self.velocity_model = np.zeros(
            (self.east_coords.size, self.north_coords.size, self.depth_coords.size)
        )

    def trim_to_octree(self, arrival_times: np.ndarray, octree: Octree) -> np.ndarray:
        east_idx = np.where(
            (self.east_coords >= octree.east_bounds[0])
            & (self.east_coords <= octree.east_bounds[1])
        )[0]
        north_idx = np.where(
            (self.north_coords >= octree.north_bounds[0])
            & (self.north_coords <= octree.north_bounds[1])
        )[0]
        depth_idx = np.where(
            (self.depth_coords >= octree.effective_depth_bounds[0])
            & (self.depth_coords <= octree.effective_depth_bounds[1])
        )[0]

        # TODO trim coords and bounds
        return arrival_times[east_idx, north_idx, depth_idx]

    def get_time_grid(self) -> np.ndarray:
        times = np.full_like(self.velocity_model, -1.0)

        station_offset = self.station.offset_to(self.origin)
        east_idx = np.argmin(np.abs(self.east_coords - station_offset[0]))
        north_idx = np.argmin(np.abs(self.north_coords - station_offset[1]))
        depth_idx = np.argmin(np.abs(self.depth_coords - station_offset[2]))
        times[east_idx, north_idx, depth_idx] = 0.0
        return times

    def interpolate_node(self, node):
        ...

    def save(self, file: Path):
        ...

    @classmethod
    def load(cls, file: Path):
        ...


class VelocityModelFactory(BaseModel):
    model: Literal["VelocityModelFactory"] = "VelocityModelFactory"

    def _get_model_grid(
        self, octree: Octree, station: Station
    ) -> StationVelocityArrivalGrid3D:
        station_offset = station.offset_to(octree.center_location)
        east_bounds = (
            min(station_offset[0], octree.east_bounds[0]),
            max(station_offset[0], octree.east_bounds[1]),
        )
        north_bounds = (
            min(station_offset[1], octree.north_bounds[0]),
            max(station_offset[1], octree.north_bounds[1]),
        )
        depth_bounds = (
            min(station_offset[2], octree.effective_depth_bounds[0]),
            max(station_offset[2], octree.effective_depth_bounds[1]),
        )

        grid_spacing = octree.smallest_node_size()

        logger.debug("Generated velocity model grid of size %s", grid.shape)
        return StationVelocityArrivalGrid3D(
            origin=octree.center_location,
            station=station,
            grid_spacing=grid_spacing,
            east_bounds=east_bounds,
            north_bounds=north_bounds,
            depth_bounds=depth_bounds,
        )

    def get_model(
        self, octree: Octree, stations: Station
    ) -> StationVelocityArrivalGrid3D:
        raise NotImplementedError


class Constant3DVelocityModel(VelocityModelFactory):
    model: Literal["Constant3DVelocityModel"] = "Constant3DVelocityModel"

    velocity: float = 5.0

    def get_model(
        self, octree: Octree, station: Station
    ) -> StationVelocityArrivalGrid3D:
        grid = self._get_model_grid(octree, station)
        grid.velocity_model.fill(self.velocity)
        return grid


VelocityModels = Annotated[
    Union[Constant3DVelocityModel, VelocityModelFactory],
    Field(..., discriminator="model"),
]
