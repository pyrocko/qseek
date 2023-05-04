from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from pydantic import BaseModel, constr, root_validator
from pyrocko.io.stationxml import load_xml
from pyrocko.model import load_stations

if TYPE_CHECKING:
    from pyrocko.model import Station as PyrockoStation

from lassie.models.location import Location

NSL_RE = r"^[a-zA-Z0-9]{0,2}\.[a-zA-Z0-9]{0,5}\.[a-zA-Z0-9]{0,3}$"

logger = logging.getLogger(__name__)


class Station(Location):
    nsl: tuple[str, str, str]

    def __hash__(self) -> int:
        return super().__hash__() + hash(self.nsl)

    @classmethod
    def from_pyrocko_station(cls, station: PyrockoStation) -> Station:
        return cls(
            nsl=station.nsl(),
            lat=station.lat,
            lon=station.lon,
            east_shift=station.east_shift,
            north_shift=station.north_shift,
            elevation=station.elevation,
            depth=station.depth,
        )


class Stations(BaseModel):
    stations: list[Station] = []
    blacklist: list[constr(regex=NSL_RE)] = []

    station_xmls: list[Path] = []
    pyrocko_station_yamls: list[Path] = []

    def __iter__(self) -> Iterator[Station]:
        for sta in self.stations:
            if ".".join(sta.nsl) in self.blacklist:
                continue
            if not sta.lat or not sta.lon:
                logger.warning("station has bad geographical coordinates: %s", sta)
                self.blacklist.append(".".join(sta.nsl))
                continue
            yield sta

    @property
    def n_stations(self) -> int:
        return len([sta for sta in self])

    def get_all_nsl(self) -> tuple[tuple[str, str, str]]:
        return tuple(recv.nsl for recv in self)

    def select(self, nsls: Iterable[tuple[str, str, str]]) -> Stations:
        """Select stations by NSL code.

        Args:
            selection (Iterable[tuple[str, str, str]]): NSL codes

        Returns:
            Stations: Containing only selected stations.
        """
        stations = []
        for nsl in nsls:
            for sta in self:
                if sta.nsl == nsl:
                    stations.append(sta)
        return Stations.construct(stations=stations)

    def get_centroid(self) -> Location:
        """Get centroid location from all stations.

        Returns:
            Location: Centroid Location.
        """
        centroid_lat, centroid_lon, centroid_elevation = np.mean(
            [(*sta.effective_lat_lon, sta.elevation) for sta in self],
            axis=0,
        )
        return Location(
            lat=centroid_lat,
            lon=centroid_lon,
            elevation=centroid_elevation,
        )

    @root_validator
    def _load_stations(cls, values) -> Any:  # noqa: N805
        loaded_stations = []
        for path in values.get("pyrocko_station_yamls"):
            loaded_stations += load_stations(filename=str(path))

        for path in values.get("station_xmls"):
            station_xml = load_xml(filename=str(path))
            loaded_stations += station_xml.get_pyrocko_stations(path)

        values.get("stations").extend(
            [Station.from_pyrocko_station(sta) for sta in loaded_stations]
        )
        if not values.get("stations"):
            raise ValueError("no stations set")
        return values
