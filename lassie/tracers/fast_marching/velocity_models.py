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

    north_coords: np.ndarray = PrivateAttr(None)
    east_coords: np.ndarray = PrivateAttr(None)
    depth_coords: np.ndarray = PrivateAttr(None)

    velocity_model: np.ndarray = PrivateAttr(None)

    _hash: str | None = PrivateAttr(None)

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
            (
                self.east_coords.size,
                self.north_coords.size,
                self.depth_coords.size,
            )
        )

    def hash(self) -> str:
        if self._hash is None:
            sha1_hash = sha1(self.velocity_model.tobytes())
            # Not needed as origin is already included in the velocity model?
            # model_hash.update(hash(self.origin).to_bytes())
            self._hash = sha1_hash.hexdigest()
        return self._hash

    def get_source_arrival_grid(self, station: Station) -> np.ndarray:
        times = np.full_like(self.velocity_model, fill_value=-1.0)

        station_offset = station.offset_to(self.origin)
        east_idx = np.argmin(np.abs(self.east_coords - station_offset[0]))
        north_idx = np.argmin(np.abs(self.north_coords - station_offset[1]))
        depth_idx = np.argmin(np.abs(self.depth_coords - station_offset[2]))
        times[east_idx, north_idx, depth_idx] = 0.0
        return times


class VelocityModelFactory(BaseModel):
    model: Literal["VelocityModelFactory"] = "VelocityModelFactory"

    grid_spacing: PositiveFloat | Literal["auto"] = Field(
        "auto",
        description="Grid spacing in meters. If 'auto' defaults to smallest octreee",
    )

    def get_model(self, octree: Octree, stations: Stations) -> VelocityModel3D:
        raise NotImplementedError


class Constant3DVelocityModel(VelocityModelFactory):
    model: Literal["Constant3DVelocityModel"] = "Constant3DVelocityModel"

    velocity: float = 5.0

    def get_model(self, octree: Octree, stations: Stations) -> VelocityModel3D:
        if self.grid_spacing == "auto":
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
        model.velocity_model.fill(self.velocity)

        return model


VelocityModels = Annotated[
    Union[Constant3DVelocityModel, VelocityModelFactory],
    Field(..., discriminator="model"),
]
