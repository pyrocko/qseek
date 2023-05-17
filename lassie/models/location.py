from __future__ import annotations

import math
from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel
from pyrocko import orthodrome as od

CoordSystem = Literal["cartesian", "geographic"]


class Location(BaseModel):
    lat: float
    lon: float
    east_shift: float = 0.0
    north_shift: float = 0.0
    elevation: float = 0.0
    depth: float = 0.0

    class Config:
        keep_untouched = (cached_property,)

    @cached_property
    def effective_lat_lon(self) -> tuple[float, float]:
        """Shift-corrected lat/lon pair of the location."""
        if self.north_shift == 0.0 and self.east_shift == 0.0:
            return self.lat, self.lon
        else:
            return od.ne_to_latlon(
                self.lat, self.lon, self.north_shift, self.east_shift
            )

    @property
    def effective_elevation(self) -> float:
        return self.elevation - self.depth

    @property
    def effective_depth(self) -> float:
        return self.depth + self.elevation

    def _same_origin(self, other: Location) -> bool:
        return bool(self.lat == other.lat and self.lon == other.lon)

    def surface_distance_to(self, other: Location) -> float:
        """Compute surface distance [m] to other location object."""

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
        if self._same_origin(other):
            return math.sqrt(
                (self.north_shift - other.north_shift) ** 2
                + (self.east_shift - other.east_shift) ** 2
                + (self.effective_elevation - other.effective_elevation) ** 2
            )

        else:
            sx, sy, sz = od.geodetic_to_ecef(
                *self.effective_lat_lon, self.effective_elevation
            )
            ox, oy, oz = od.geodetic_to_ecef(
                *other.effective_lat_lon, other.effective_elevation
            )

            return math.sqrt((sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2)

    def __hash__(self) -> int:
        return hash(
            (
                self.lat,
                self.lon,
                self.east_shift,
                self.north_shift,
                self.elevation,
                self.depth,
            )
        )


def locations_to_csv(locations: Iterable[Location], filename: Path) -> Path:
    lines = ["lat, lon, elevation, type"]
    for loc in locations:
        lines.append(
            "%.4f, %.4f, %.4f, %s"
            % (*loc.effective_lat_lon, loc.effective_elevation, loc.__class__.__name__)
        )
    filename.write_text("\n".join(lines))
    return filename
