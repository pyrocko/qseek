from __future__ import annotations

import hashlib
import math
import struct
from typing import TYPE_CHECKING, Iterable, Literal, TypeVar

from pydantic import BaseModel, PrivateAttr
from pyrocko import orthodrome as od
from typing_extensions import Self

if TYPE_CHECKING:
    from pathlib import Path

CoordSystem = Literal["cartesian", "geographic"]


class Location(BaseModel):
    lat: float
    lon: float
    east_shift: float = 0.0
    north_shift: float = 0.0
    elevation: float = 0.0
    depth: float = 0.0

    _cached_lat_lon: tuple[float, float] | None = PrivateAttr(None)

    @property
    def effective_lat(self) -> float:
        return self.effective_lat_lon[0]

    @property
    def effective_lon(self) -> float:
        return self.effective_lat_lon[1]

    @property
    def effective_lat_lon(self) -> tuple[float, float]:
        """Shift-corrected lat/lon pair of the location."""
        if self._cached_lat_lon is None:
            if self.north_shift == 0.0 and self.east_shift == 0.0:
                self._cached_lat_lon = self.lat, self.lon
            else:
                lat, lon = od.ne_to_latlon(
                    self.lat,
                    self.lon,
                    self.north_shift,
                    self.east_shift,
                )
                self._cached_lat_lon = float(lat), float(lon)
        return self._cached_lat_lon

    @property
    def effective_elevation(self) -> float:
        return self.elevation - self.depth

    @property
    def effective_depth(self) -> float:
        return self.depth - self.elevation

    def _same_origin(self, other: Location) -> bool:
        return bool(self.lat == other.lat and self.lon == other.lon)

    def surface_distance_to(self, other: Location) -> float:
        """Compute surface distance [m] to other location object.

        Args:
            other (Location): The other location.

        Returns:
            float: The surface distance in [m].
        """

        if self._same_origin(other):
            return math.sqrt(
                (self.north_shift - other.north_shift) ** 2
                + (self.east_shift - other.east_shift) ** 2
            )
        return float(
            od.distance_accurate50m_numpy(
                *self.effective_lat_lon, *other.effective_lat_lon
            )[0]
        )

    def distance_to(self, other: Location) -> float:
        """Compute 3-dimensional distance [m] to other location object.

        Args:
            other (Location): The other location.

        Returns:
            float: The distance in [m].
        """
        if self._same_origin(other):
            return math.sqrt(
                (self.north_shift - other.north_shift) ** 2
                + (self.east_shift - other.east_shift) ** 2
                + (self.effective_elevation - other.effective_elevation) ** 2
            )

        sx, sy, sz = od.geodetic_to_ecef(
            *self.effective_lat_lon, self.effective_elevation
        )
        ox, oy, oz = od.geodetic_to_ecef(
            *other.effective_lat_lon, other.effective_elevation
        )

        return math.sqrt((sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2)

    def offset_from(self, other: Location) -> tuple[float, float, float]:
        """Return offset vector (east, north, depth) to other location in [m]

        Args:
            other (Location): The other location.

        Returns:
            tuple[float, float, float]: The offset vector.
        """
        if self._same_origin(other):
            return (
                self.east_shift - other.east_shift,
                self.north_shift - other.north_shift,
                -(self.effective_elevation - other.effective_elevation),
            )

        shift_north, shift_east = od.latlon_to_ne_numpy(
            self.lat, self.lon, other.lat, other.lon
        )

        return (
            self.east_shift - other.east_shift - shift_east[0],
            self.north_shift - other.north_shift - shift_north[0],
            -(self.effective_elevation - other.effective_elevation),
        )

    def shifted_origin(self) -> Self:
        """Shift the origin of the location to the effective lat/lon.

        Returns:
            Self: The shifted location.
        """
        shifted = self.model_copy()
        shifted.lat = self.effective_lat
        shifted.lon = self.effective_lon
        shifted.east_shift = 0.0
        shifted.north_shift = 0.0
        return shifted

    def __hash__(self) -> int:
        return hash(self.location_hash())

    def location_hash(self) -> str:
        sha1 = hashlib.sha1(
            struct.pack(
                "dddddd",
                self.lat,
                self.lon,
                self.east_shift,
                self.north_shift,
                self.elevation,
                self.depth,
            )
        )
        return sha1.hexdigest()


def locations_to_csv(locations: Iterable[Location], filename: Path) -> Path:
    lines = ["lat, lon, elevation, type"]
    for loc in locations:
        lines.append(
            "%.4f, %.4f, %.4f, %s"
            % (*loc.effective_lat_lon, loc.effective_elevation, loc.__class__.__name__)
        )
    filename.write_text("\n".join(lines))
    return filename


LocationType = TypeVar("LocationType", bound=Location)
