from __future__ import annotations

import logging
import re
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, Sequence, Union, cast

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PrivateAttr,
)
from pydantic.dataclasses import dataclass
from pyevtk.hl import gridToVTK
from pyrocko import orthodrome as od
from pyrocko.cake import LayeredModel, load_model
from scipy.interpolate import RegularGridInterpolator

from qseek.base import Model
from qseek.models.location import Location
from qseek.types import FilePath

if TYPE_CHECKING:
    from qseek.models.station import StationInventory
    from qseek.octree import Node, Octree


KM = 1e3
logger = logging.getLogger(__name__)

# Version of the cached travel time volumes, part of the model hash. Increase it
# when a change of the code changes the volumes of the same model and stations.
VOLUME_VERSION = 2


def grid_offsets(locations: Sequence[Location], center: Location) -> np.ndarray:
    """Offsets of locations in the grid frame of a center location.

    The grid is laid out in east and north shifts from the geographic origin of
    `center` (its lat/lon). Locations with the same origin are placed by their
    shifts; other locations by their bearing and distance on the WGS84 ellipsoid
    from that origin, the distances qseek's other tracers use. The bearing is
    measured at the center: `Location.offset_from` measures it at the location,
    which rotates far locations by the meridian convergence.

    Args:
        locations (Sequence[Location]): The locations.
        center (Location): The center location of the grid.

    Returns:
        np.ndarray: Offsets (east, north, depth) in [m], shape (N, 3).
    """
    offsets = np.empty((len(locations), 3))
    if not len(locations):
        return offsets
    same_origin = np.array(
        [loc.lat == center.lat and loc.lon == center.lon for loc in locations]
    )
    shifts = np.array([(loc.east_shift, loc.north_shift) for loc in locations])
    offsets[same_origin, 0] = shifts[same_origin, 0] - center.east_shift
    offsets[same_origin, 1] = shifts[same_origin, 1] - center.north_shift
    if not same_origin.all():
        # From the raw coordinates: a copy with updated shifts keeps the cached
        # effective_lat_lon of the original
        lat_lon = np.array(
            [
                (loc.lat, loc.lon)
                if loc.north_shift == 0.0 and loc.east_shift == 0.0
                else od.ne_to_latlon(loc.lat, loc.lon, loc.north_shift, loc.east_shift)
                for loc, same in zip(locations, same_origin, strict=True)
                if not same
            ],
            dtype=float,
        )
        north, east = od.latlon_to_ne_numpy(
            center.lat, center.lon, lat_lon[:, 0], lat_lon[:, 1]
        )
        offsets[~same_origin, 0] = east - center.east_shift
        offsets[~same_origin, 1] = north - center.north_shift
    offsets[:, 2] = [loc.effective_depth - center.effective_depth for loc in locations]
    return offsets


def snap_bounds(
    bounds: tuple[float, float], grid_spacing: float, outward: bool
) -> tuple[float, float]:
    """Snap the end of bounds to a whole number of grid cells from the start.

    Args:
        bounds (tuple[float, float]): Start and end in [m].
        grid_spacing (float): Grid spacing in [m].
        outward (bool): Extend the bounds to cover them, else shrink them to stay
            inside.

    Returns:
        tuple[float, float]: The snapped bounds.
    """
    n_cells = (bounds[1] - bounds[0]) / grid_spacing
    n_cells = np.ceil(n_cells - 1e-6) if outward else np.floor(n_cells + 1e-6)
    return float(bounds[0]), float(bounds[0] + n_cells * grid_spacing)


def grid_offset(location: Location, center: Location) -> np.ndarray:
    """Offset (east, north, depth) of one location in the grid frame, in [m]."""
    return grid_offsets([location], center)[0]


def node_grid_offsets(nodes: Sequence[Node], center: Location) -> np.ndarray:
    """Offsets of octree nodes in the grid frame of a center location.

    Same as `grid_offsets` of the node locations, vectorized over the nodes of
    one octree.

    Args:
        nodes (Sequence[Node]): Nodes of one octree.
        center (Location): The center location of the grid.

    Returns:
        np.ndarray: Offsets (east, north, depth) in [m], shape (N, 3).
    """
    if not nodes or any(node.tree is not nodes[0].tree for node in nodes):
        return grid_offsets([node.as_location() for node in nodes], center)

    reference = nodes[0].tree.location
    node_offsets = np.array([(node.east, node.north, node.depth) for node in nodes])
    east = reference.east_shift + node_offsets[:, 0]
    north = reference.north_shift + node_offsets[:, 1]
    offsets = np.empty_like(node_offsets)
    if reference.lat == center.lat and reference.lon == center.lon:
        offsets[:, 0] = east - center.east_shift
        offsets[:, 1] = north - center.north_shift
    else:
        lat, lon = od.ne_to_latlon(reference.lat, reference.lon, north, east)
        north, east = od.latlon_to_ne_numpy(center.lat, center.lon, lat, lon)
        offsets[:, 0] = east - center.east_shift
        offsets[:, 1] = north - center.north_shift
    offsets[:, 2] = (
        reference.effective_depth + node_offsets[:, 2] - center.effective_depth
    )
    return offsets


class VelocityModel3D(BaseModel):
    center: Location

    grid_spacing: float

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]

    _east_coords: np.ndarray = PrivateAttr()
    _north_coords: np.ndarray = PrivateAttr()
    _depth_coords: np.ndarray = PrivateAttr()

    _velocity_model: np.ndarray = PrivateAttr()

    _hash: str | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing
        for name, size in (
            ("east", self.east_size),
            ("north", self.north_size),
            ("depth", self.depth_size),
        ):
            n_cells = size / grid_spacing
            if abs(n_cells - round(n_cells)) > 1e-6:
                raise ValueError(
                    f"{name} extent {size} m is not a multiple of the grid spacing "
                    f"{grid_spacing} m; the eikonal solver would use another spacing "
                    "than the grid coordinates"
                )
        neast = round(self.east_size / grid_spacing) + 1
        nnorth = round(self.north_size / grid_spacing) + 1
        ndepth = round(self.depth_size / grid_spacing) + 1

        self._east_coords = np.linspace(
            self.east_bounds[0],
            self.east_bounds[1],
            neast,
            endpoint=True,
        )
        self._north_coords = np.linspace(
            self.north_bounds[0],
            self.north_bounds[1],
            nnorth,
            endpoint=True,
        )
        self._depth_coords = np.linspace(
            self.depth_bounds[0],
            self.depth_bounds[1],
            ndepth,
            endpoint=True,
        )

        self._velocity_model = np.zeros(
            (
                self._east_coords.size,
                self._north_coords.size,
                self._depth_coords.size,
            )
        )

    def set_velocity_model(self, velocity_model: np.ndarray) -> None:
        if velocity_model.shape != self._velocity_model.shape:
            raise ValueError(
                f"Velocity model shape {velocity_model.shape} does not match"
                f" expected shape {self._velocity_model.shape}"
            )
        self._velocity_model = velocity_model.astype(float, copy=False)

    @property
    def velocity_model(self) -> np.ndarray:
        return self._velocity_model

    @property
    def east_coords(self) -> np.ndarray:
        return self._east_coords

    @property
    def north_coords(self) -> np.ndarray:
        return self._north_coords

    @property
    def depth_coords(self) -> np.ndarray:
        return self._depth_coords

    @property
    def east_size(self) -> float:
        return self.east_bounds[1] - self.east_bounds[0]

    @property
    def north_size(self) -> float:
        return self.north_bounds[1] - self.north_bounds[0]

    @property
    def depth_size(self) -> float:
        return self.depth_bounds[1] - self.depth_bounds[0]

    @property
    def volume(self) -> float:
        return self.east_size * self.north_size * self.depth_size

    def hash(self) -> str:
        """Return hash of velocity model.

        Returns:
            str: The hash.
        """
        if self._hash is None:
            hash = sha1(self._velocity_model.tobytes())
            hash.update(self.center.location_hash().encode())
            # Models with the same values on another grid need other volumes
            geometry = (
                self.grid_spacing,
                *self.east_bounds,
                *self.north_bounds,
                *self.depth_bounds,
                *self._velocity_model.shape,
            )
            hash.update(np.asarray(geometry, dtype=float).tobytes())
            hash.update(f"volume-version-{VOLUME_VERSION}".encode())
            self._hash = hash.hexdigest()
        return self._hash

    def check_velocities(self, label: str = "") -> None:
        """Raise if the velocity model has non-finite or non-positive velocities.

        Args:
            label (str, optional): Name of the model for the error message.
        """
        bad = ~(np.isfinite(self._velocity_model) & (self._velocity_model > 0.0))
        if not bad.any():
            return
        east_idx, north_idx, depth_idx = np.nonzero(bad)
        raise ValueError(
            f"velocity model {label} has {bad.sum()} cells with zero, negative or "
            "non-finite velocities, e.g. a SLOW_LEN of zero; the eikonal solver "
            "needs positive velocities. Cells at east "
            f"{self._east_coords[east_idx].min():.0f} to "
            f"{self._east_coords[east_idx].max():.0f} m, north "
            f"{self._north_coords[north_idx].min():.0f} to "
            f"{self._north_coords[north_idx].max():.0f} m, depth "
            f"{self._depth_coords[depth_idx].min():.0f} to "
            f"{self._depth_coords[depth_idx].max():.0f} m from the center."
        )

    def get_location_indices(
        self,
        location: Location,
        raise_exc: bool = True,
    ) -> tuple[int, int, int]:
        """Return indices of location in velocity model, by nearest neighbor.

        Args:
            location (Location): The location.
            raise_exc (bool, optional): Raise exception if location is outside of

        Returns:
            tuple[int, int, int]: The indices as (east, north, depth).
        """
        if raise_exc and not self.is_inside(location):
            raise ValueError("Location is outside of velocity model.")
        station_offset = grid_offset(location, self.center)
        east_idx = np.argmin(np.abs(self._east_coords - station_offset[0]))
        north_idx = np.argmin(np.abs(self._north_coords - station_offset[1]))
        depth_idx = np.argmin(np.abs(self._depth_coords - station_offset[2]))
        return round(east_idx), round(north_idx), round(depth_idx)

    def get_velocity(self, location: Location) -> float:
        """Return velocity at location in [m/s], nearest neighbor.

        Args:
            location (Location): The location.

        Returns:
            float: The velocity in m/s.
        """
        east_idx, north_idx, depth_idx = self.get_location_indices(location)
        return self.velocity_model[east_idx, north_idx, depth_idx]

    def crop_to_stations_and_octree(
        self,
        stations: StationInventory,
        octree: Octree,
        margin: int = 4,
    ) -> VelocityModel3D:
        """Crop velocity model to stations and octree.

        Args:
            stations (StationInventory): The stations.
            octree (Octree): The octree.
            margin (int, optional): Margin nodes around stations and octree corners.
                Defaults to 4.

        Returns:
            Self: The cropped velocity model.
        """
        return self.crop_to_locations(
            list(stations) + octree.get_corners(),
            margin=margin,
        )

    def crop_to_locations(
        self, locations: list[Location], margin: int = 4
    ) -> VelocityModel3D:
        """Crop velocity model to locations.

        Args:
            locations (list[Location]): The locations.
            margin (int, optional): Margin nodes around locations. Defaults to 4.

        Returns:
            Self: The cropped velocity model.
        """
        logger.info("cropping velocity model to ROI volume")
        east_idx, north_idx, depth_idx = zip(
            *[
                self.get_location_indices(location, raise_exc=False)
                for location in locations
            ],
            strict=True,
        )

        east_min, east_max = (
            max(0, min(east_idx) - margin),
            min(max(east_idx) + margin, self._east_coords.size - 1),
        )

        north_min, north_max = (
            max(0, min(north_idx) - margin),
            min(max(north_idx) + margin, self._north_coords.size - 1),
        )
        depth_min, depth_max = (
            max(0, min(depth_idx) - margin),
            min(max(depth_idx) + margin, self._depth_coords.size - 1),
        )

        cropped_model = VelocityModel3D(
            center=self.center,
            grid_spacing=self.grid_spacing,
            east_bounds=(
                float(self._east_coords[east_min]),
                float(self._east_coords[east_max]),
            ),
            north_bounds=(
                float(self._north_coords[north_min]),
                float(self._north_coords[north_max]),
            ),
            depth_bounds=(
                float(self._depth_coords[depth_min]),
                float(self._depth_coords[depth_max]),
            ),
        )
        cropped_model.set_velocity_model(
            self._velocity_model[
                east_min : east_max + 1,
                north_min : north_max + 1,
                depth_min : depth_max + 1,
            ].copy()
        )
        logger.info(
            "reduce velocity model volume by %.1f%%: Old volume %.2f km³, new %.2f km³",
            (1.0 - cropped_model.volume / self.volume) * 100,
            self.volume / KM**3,
            cropped_model.volume / KM**3,
        )
        return cropped_model

    def is_inside(self, location: Location) -> bool:
        """Return True if location is inside velocity model.

        Args:
            location (Location): The location.

        Returns:
            bool: True if location is inside velocity model.
        """
        offset = grid_offset(location, self.center)
        # The same tolerance as the interpolation of the travel time volumes, so
        # that locations on the edge are not lost to rounding
        tolerance = 1e-6 * self.grid_spacing
        return all(
            start - tolerance <= value <= end + tolerance
            for value, (start, end) in zip(
                offset,
                (self.east_bounds, self.north_bounds, self.depth_bounds),
                strict=True,
            )
        )

    def get_meshgrid(self) -> tuple[np.ndarray, ...]:
        """Return meshgrid of velocity model coordinates.

        Returns:
            tuple[np.ndarray, ...]: The meshgrid as tuple of numpy arrays for east, north,
                depth.
        """
        return np.meshgrid(
            self._east_coords,
            self._north_coords,
            self._depth_coords,
            indexing="ij",
        )

    def get_source_distance_grid(self, source: Location) -> np.ndarray:
        """Return a 3D grid of eukledian distances to a source.

        Args:
            source (Location): The source location.

        Returns:
            np.ndarray: The 3D grid of distances to the source.
        """
        east, north, depth = self.get_meshgrid()
        source_offset = grid_offset(source, self.center)
        return np.sqrt(
            (east - source_offset[0]) ** 2
            + (north - source_offset[1]) ** 2
            + (depth - source_offset[2]) ** 2
        )

    def resample(
        self,
        grid_spacing: float,
        method: Literal["nearest", "linear", "cubic"] = "nearest",
    ) -> VelocityModel3D:
        """Resample velocity model to new grid spacing.

        Args:
            grid_spacing (float): The new grid spacing in meters.
            method (Literal['nearest', 'linear', 'cubic'], optional): Interpolation
                method. Defaults to "linear".

        Returns:
            Self: A new, resampled velocity model.
        """
        if grid_spacing == self.grid_spacing:
            return self

        logger.info(
            "resampling velocity model to grid spacing %s m using %s interpolation",
            grid_spacing,
            method,
        )
        interpolator = RegularGridInterpolator(
            (self._east_coords, self._north_coords, self._depth_coords),
            self._velocity_model,
            method=method,
            bounds_error=True,
        )
        # Whole cells of the new spacing, inside the model
        resampled_model = VelocityModel3D(
            center=self.center,
            grid_spacing=grid_spacing,
            east_bounds=snap_bounds(self.east_bounds, grid_spacing, outward=False),
            north_bounds=snap_bounds(self.north_bounds, grid_spacing, outward=False),
            depth_bounds=snap_bounds(self.depth_bounds, grid_spacing, outward=False),
        )
        coordinates = np.array(
            [coords.ravel() for coords in resampled_model.get_meshgrid()],
        ).T

        velocity_model = (
            interpolator(coordinates)
            .astype(dtype=np.float32, copy=False)
            .reshape(resampled_model._velocity_model.shape)
        )
        resampled_model.set_velocity_model(velocity_model)
        return resampled_model

    def export_vtk(self, filename: Path, reference: Location | None = None) -> None:
        offset = grid_offset(reference, self.center) if reference else np.zeros(3)

        out_file = gridToVTK(
            str(filename),
            self._east_coords - offset[0],
            self._north_coords - offset[1],
            np.array((-self._depth_coords + offset[2])[::-1]),
            pointData={"velocity": np.array(self._velocity_model[:, :, ::-1])},
        )
        logger.info("exported vtk velocity model to %s", out_file)


class VelocityModelFactory(Model):
    model: Literal["VelocityModelFactory"] = "VelocityModelFactory"

    grid_spacing: PositiveFloat | Literal["octree"] = Field(
        default="octree",
        description="Grid spacing in meters."
        " If 'octree' defaults to smallest octreee node size.",
    )

    def prepare(self) -> None:
        pass

    def get_model(
        self, octree: Octree, stations: StationInventory | None = None
    ) -> VelocityModel3D:
        raise NotImplementedError


class Constant3DVelocityModel(VelocityModelFactory):
    """This model is for mere testing of the method."""

    model: Literal["Constant3DVelocityModel"] = "Constant3DVelocityModel"

    velocity: PositiveFloat = 5000.0

    def get_model(
        self, octree: Octree, stations: StationInventory | None = None
    ) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        else:
            grid_spacing = self.grid_spacing

        # Whole cells, covering the octree
        model = VelocityModel3D(
            center=octree.location,
            grid_spacing=grid_spacing,
            east_bounds=snap_bounds(octree.east_bounds, grid_spacing, outward=True),
            north_bounds=snap_bounds(octree.north_bounds, grid_spacing, outward=True),
            depth_bounds=snap_bounds(octree.depth_bounds, grid_spacing, outward=True),
        )
        model._velocity_model.fill(self.velocity)

        return model


NonLinLocGridType = Literal["VELOCITY", "VELOCITY_METERS", "SLOW_LEN"]
GridDtype = Literal["FLOAT", "DOUBLE"]
DTYPE_MAP: dict[GridDtype, type] = {"FLOAT": np.float32, "DOUBLE": float}


@dataclass
class NonLinLocHeader:
    """Helper class representing a NonLinLoc header file."""

    origin: Location  # Reference location of the model
    nx: int
    ny: int
    nz: int
    delta_x: float
    delta_y: float
    delta_z: float
    grid_dtype: GridDtype
    grid_type: NonLinLocGridType

    @classmethod
    def from_header_file(
        cls,
        file: Path,
        reference_location: Location | None = None,
    ) -> Self:
        """Load NonLinLoc velocity model header file.

        Args:
            file (Path): Path to NonLinLoc model header file.
            reference_location (Location | None, optional): Origin of the
                NonLinLoc model's coordinates. Required.

        Raises:
            ValueError: If the reference location is missing, the grid spacing
                is not equal in all dimensions, or the transformation is not
                supported.

        Returns:
            Self: The header.
        """
        logger.info("loading NonLinLoc velocity model header file %s", file)
        try:
            header_lines = file.read_text().split("\n")
        except Exception as e:
            raise ValueError(f"Error reading NonLinLoc header file {file}") from e
        header_text = header_lines[0]
        cls._check_transform(file, header_lines[1] if len(header_lines) > 1 else "")
        header_text = re.sub(r"\s+", " ", header_text)  # remove excessive spaces

        parts = header_text.split()
        n_parts = len(parts)
        (
            nx,
            ny,
            nz,
            orig_x,
            orig_y,
            orig_z,
            delta_x,
            delta_y,
            delta_z,
            grid_type,
            *_,
        ) = parts
        if n_parts == 10:
            logger.warning(
                "grid dtype not specified in NonLinLoc header, defaulting to FLOAT"
            )
            grid_dtype = "FLOAT"
        elif n_parts == 11:
            grid_dtype = parts[10]
        else:
            raise ValueError("Invalid NonLinLoc header format")

        if not float(delta_x) == float(delta_y) == float(delta_z):
            raise ValueError("NonLinLoc velocity model must have equal spacing.")

        if reference_location is None:
            raise ValueError(
                "NonLinLoc grids are in km relative to the origin of their"
                " transformation, which the header does not hold: set"
                " `reference_location` to that origin"
            )
        origin = reference_location.model_copy(deep=True)
        origin.east_shift += float(orig_x) * KM
        origin.north_shift += float(orig_y) * KM
        origin.elevation -= float(orig_z) * KM
        logger.info("Shifted NonLinLoc reference location: %s", origin)

        return cls(
            origin=origin,
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            delta_x=float(delta_x) * KM,
            delta_y=float(delta_y) * KM,
            delta_z=float(delta_z) * KM,
            grid_dtype=cast(GridDtype, grid_dtype),
            grid_type=cast(NonLinLocGridType, grid_type),
        )

    @staticmethod
    def _check_transform(file: Path, line: str) -> None:
        """Reject transformations the grid frame cannot represent."""
        parts = line.split()
        if not parts or parts[0] != "TRANSFORM":
            return
        kind = parts[1] if len(parts) > 1 else "NONE"
        if kind == "GLOBAL":
            raise ValueError(
                f"NonLinLoc header {file}: GLOBAL grids in degrees are not supported"
            )
        if "RotCW" in parts:
            rotation = float(parts[parts.index("RotCW") + 1])
            if rotation != 0.0:
                raise ValueError(
                    f"NonLinLoc header {file}: rotated grids (RotCW {rotation}) are"
                    " not supported"
                )
        if kind not in ("NONE", "SIMPLE"):
            logger.warning(
                "NonLinLoc header %s: transformation %s is approximated by an"
                " azimuthal projection around the reference location",
                file,
                kind,
            )

    @property
    def dtype(self) -> type:
        """Dtype of the grid."""
        return DTYPE_MAP[self.grid_dtype]

    @property
    def grid_spacing(self) -> float:
        """Grid spacing, homogeneous in three directions."""
        return self.delta_x

    @property
    def east_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        size = self.delta_x * (self.nx - 1)
        return -size / 2, size / 2

    @property
    def north_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        size = self.delta_y * (self.ny - 1)
        return -size / 2, size / 2

    @property
    def depth_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        return 0.0, self.delta_z * (self.nz - 1)

    @property
    def center(self) -> Location:
        """Return center location of velocity model.

        Returns:
            Location: The center location of the grid.
        """
        center = self.origin.model_copy(deep=True)
        center.east_shift += self.delta_x * (self.nx - 1) / 2
        center.north_shift += self.delta_y * (self.ny - 1) / 2
        return center


class NonLinLocVelocityModel(VelocityModelFactory):
    model: Literal["NonLinLocVelocityModel"] = "NonLinLocVelocityModel"

    header_file: FilePath = Field(
        ...,
        description="Path to NonLinLoc model header file file. "
        "The file should be in the format of a NonLinLoc velocity model header file. "
        "Binary data has to have the same name and `.buf` suffix.",
    )
    grid_spacing: PositiveFloat | Literal["octree", "input"] = Field(
        default="input",
        description="Grid spacing in meters. "
        "If 'octree' defaults to smallest octreee node size. If 'input' uses the"
        " grid spacing from the NonLinLoc header file.",
    )
    interpolation: Literal["nearest", "linear", "cubic"] = Field(
        default="cubic",
        description="Interpolation method for resampling the grid"
        " for the fast-marching method. Choose from `nearest`, `linear` or `cubic`.",
    )
    reference_location: Location | None = Field(
        default=None,
        description="Geographic origin of the NonLinLoc model's coordinates"
        " (required). The grid's origin from the header is applied as shifts to"
        " this location; NonLinLoc's z is relative to its elevation.",
    )

    _header: NonLinLocHeader = PrivateAttr()
    _velocity_model: np.ndarray = PrivateAttr()

    def prepare(self):
        self._header = NonLinLocHeader.from_header_file(
            self.header_file,
            reference_location=self.reference_location,
        )
        header = self._header
        buffer_file = self.header_file.with_suffix(".buf")
        if not buffer_file.exists():
            raise FileNotFoundError(f"Buffer file {buffer_file} not found.")

        logger.debug("loading NonLinLoc velocity model buffer file %s", buffer_file)
        self._velocity_model = np.fromfile(buffer_file, dtype=header.dtype).reshape(
            (header.nx, header.ny, header.nz)
        )

        if header.grid_type == "SLOW_LEN":
            logger.debug("converting NonLinLoc SLOW_LEN model to velocity")
            self._velocity_model = 1.0 / (self._velocity_model / header.grid_spacing)
        elif header.grid_type == "VELOCITY":
            self._velocity_model *= KM
        elif header.grid_type == "VELOCITY_METERS":
            pass
        else:
            raise ValueError(f"Unsupported NonLinLoc grid type {header.grid_type}")

        self._velocity_model = self._velocity_model.astype(np.float32, copy=False)
        bounds_tpl = "{0} - {1}"
        logger.info(
            "NonLinLoc velocity model %s"
            " east_bounds %s, north_bounds %s, depth_bounds %s, spacing %.2f m",
            header.center,
            bounds_tpl.format(*header.east_bounds),
            bounds_tpl.format(*header.north_bounds),
            bounds_tpl.format(*header.depth_bounds),
            header.grid_spacing,
        )
        return self

    def get_model(
        self, octree: Octree, stations: StationInventory | None = None
    ) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        elif self.grid_spacing == "input":
            grid_spacing = self._header.grid_spacing
        elif isinstance(self.grid_spacing, float):
            grid_spacing = self.grid_spacing
        else:
            raise ValueError(f"Invalid grid_spacing {self.grid_spacing}")

        header = self._header

        velocity_model = VelocityModel3D(
            center=header.center,
            grid_spacing=header.grid_spacing,
            east_bounds=header.east_bounds,
            north_bounds=header.north_bounds,
            depth_bounds=header.depth_bounds,
        )
        velocity_model.set_velocity_model(self._velocity_model)
        if stations:
            velocity_model = velocity_model.crop_to_stations_and_octree(
                stations,
                octree,
            )
        return velocity_model.resample(grid_spacing, method=self.interpolation)


class VelocityModelLayered(VelocityModelFactory):
    # For mere testing purposes of the 3D tracer against Pyrocko cake 2D travel times
    model: Literal["VelocityModel2D"] = "VelocityModel2D"
    velocity: Literal["vp", "vs"] = Field(
        default="vp",
        description="velocity to extract from the 2D model, choose from 'vp' or 'vs'.",
    )
    format: Literal["nd", "hyposat"] = Field(
        default="nd",
        description="Format of the velocity model. nd or hyposat is supported.",
    )
    filename: FilePath = Field(
        ...,
        description="Path to `.nd` file holding the 2D velocity model information.",
    )
    raw_file_data: str | None = Field(
        default=None,
        description="Raw `.nd` file data.",
    )

    _layered_model: LayeredModel = PrivateAttr()

    def prepare(self):
        if self.filename is not None:
            logger.info("loading velocity model from %s", self.filename)
            self.raw_file_data = self.filename.read_text()

        if self.raw_file_data is not None:
            with NamedTemporaryFile("w") as tmpfile:
                tmpfile.write(self.raw_file_data)
                tmpfile.flush()
                self._layered_model = load_model(
                    tmpfile.name,
                    format=self.format,
                )
        else:
            raise AttributeError("No velocity model or crust2 profile defined.")
        return self

    def get_model(
        self, octree: Octree, stations: StationInventory | None = None
    ) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        else:
            grid_spacing = self.grid_spacing

        # Whole cells, covering the octree
        model = VelocityModel3D(
            center=octree.location,
            grid_spacing=grid_spacing,
            east_bounds=snap_bounds(octree.east_bounds, grid_spacing, outward=True),
            north_bounds=snap_bounds(octree.north_bounds, grid_spacing, outward=True),
            depth_bounds=snap_bounds(octree.depth_bounds, grid_spacing, outward=True),
        )

        velocities = []
        # The layered model is in absolute depths, the grid relative to its center
        for depth in model.center.effective_depth + model.depth_coords:
            material = self._layered_model.material(z=depth)
            if self.velocity == "vp":
                velocities.append(material.vp)
            elif self.velocity == "vs":
                velocities.append(material.vs)
            else:
                raise ValueError(f"Invalid velocity {self.velocity}")

        velocities = np.array(velocities)

        model.velocity_model[:, :, :] = velocities[np.newaxis, np.newaxis, :]
        return model


VelocityModels = Annotated[
    Union[Constant3DVelocityModel, NonLinLocVelocityModel, VelocityModelLayered],
    Field(..., discriminator="model"),
]
