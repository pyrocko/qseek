from __future__ import annotations

import logging
from hashlib import sha1
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, PrivateAttr

from lassie.models.location import Location

if TYPE_CHECKING:
    from lassie.models.station import Station, Stations
    from lassie.octree import Octree


logger = logging.getLogger(__name__)


class VelocityModel3D(BaseModel):
    origin: Location

    grid_spacing: float

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]

    _north_coords: np.ndarray = PrivateAttr(None)
    _east_coords: np.ndarray = PrivateAttr(None)
    _depth_coords: np.ndarray = PrivateAttr(None)

    _velocity_model: np.ndarray = PrivateAttr(None)

    _hash: str | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self._east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self._north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self._depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1] + grid_spacing,
            grid_spacing,
        )

        self._velocity_model = np.zeros(
            (
                self._east_coords.size,
                self._north_coords.size,
                self._depth_coords.size,
            )
        )

    def hash(self) -> str:
        if self._hash is None:
            sha1_hash = sha1(self._velocity_model.tobytes())
            self._hash = sha1_hash.hexdigest()
        return self._hash

    def get_source_arrival_grid(self, station: Station) -> np.ndarray:
        times = np.full_like(self._velocity_model, fill_value=-1.0)

        station_offset = station.offset_to(self.origin)
        east_idx = np.argmin(np.abs(self._east_coords - station_offset[0]))
        north_idx = np.argmin(np.abs(self._north_coords - station_offset[1]))
        depth_idx = np.argmin(np.abs(self._depth_coords - station_offset[2]))
        times[east_idx, north_idx, depth_idx] = 0.0
        return times


class VelocityModelFactory(BaseModel):
    model: Literal["VelocityModelFactory"] = "VelocityModelFactory"

    grid_spacing: PositiveFloat | Literal["quadtree"] = Field(
        "quadtree",
        description="Grid spacing in meters."
        " If 'quadtree' defaults to smallest octreee node size.",
    )

    def get_model(self, octree: Octree, stations: Stations) -> VelocityModel3D:
        raise NotImplementedError


class Constant3DVelocityModel(VelocityModelFactory):
    model: Literal["Constant3DVelocityModel"] = "Constant3DVelocityModel"

    velocity: PositiveFloat = 5000.0

    def get_model(self, octree: Octree, stations: Stations) -> VelocityModel3D:
        if self.grid_spacing == "quadtree":
            grid_spacing = octree.smallest_node_size()
        else:
            grid_spacing = self.grid_spacing

        model = VelocityModel3D(
            origin=octree.center_location,
            grid_spacing=grid_spacing,
            east_bounds=octree.east_bounds,
            north_bounds=octree.north_bounds,
            depth_bounds=octree.effective_depth_bounds,
        )
        model._velocity_model.fill(self.velocity)

        return model


VelocityModels = Annotated[
    Union[Constant3DVelocityModel, VelocityModelFactory],
    Field(..., discriminator="model"),
]
