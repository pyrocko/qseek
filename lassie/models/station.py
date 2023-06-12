from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from pydantic import BaseModel, PrivateAttr, constr, root_validator
from pyrocko.io.stationxml import load_xml
from pyrocko.model import Station as PyrockoStation
from pyrocko.model import dump_stations_yaml, load_stations
from pathlib import Path

if TYPE_CHECKING:
    from pyrocko.trace import Trace

from lassie.models.location import CoordSystem, Location

NSL_RE = r"^[a-zA-Z0-9]{0,2}\.[a-zA-Z0-9]{0,5}\.[a-zA-Z0-9]{0,3}$"

logger = logging.getLogger(__name__)


class Station(Location):
    network: str
    station: str
    location: str

    @classmethod
    def from_pyrocko_station(cls, station: PyrockoStation) -> Station:
        return cls(
            network=station.network,
            station=station.station,
            location=station.location,
            lat=station.lat,
            lon=station.lon,
            east_shift=station.east_shift,
            north_shift=station.north_shift,
            elevation=station.elevation,
            depth=station.depth,
        )

    def to_pyrocko_station(self) -> PyrockoStation:
        return PyrockoStation(**self.dict(exclude={"effective_lat_lon"}))

    @property
    def pretty_nsl(self) -> str:
        return ".".join(self.nsl)

    @property
    def nsl(self) -> tuple[str, str, str]:
        return self.network, self.station, self.location

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.nsl))


class Stations(BaseModel):
    stations: list[Station] = []
    blacklist: set[constr(regex=NSL_RE)] = set()

    station_xmls: list[Path] = []
    pyrocko_station_yamls: list[Path] = []

    _cached_coordinates: np.ndarray | None = PrivateAttr(None)

    def __iter__(self) -> Iterator[Station]:
        for sta in self.stations:
            if sta.pretty_nsl in self.blacklist:
                continue
            yield sta

    @property
    def n_stations(self) -> int:
        return len([sta for sta in self])

    def get_all_nsl(self) -> tuple[tuple[str, str, str]]:
        return tuple(sta.nsl for sta in self)

    def select_from_traces(self, traces: Iterable[Trace]) -> Stations:
        """Select stations by NSL code.

        Args:
            selection (Iterable[tuple[str, str, str]]): NSL codes

        Returns:
            Stations: Containing only selected stations.
        """
        stations = []
        for nsl in ((tr.network, tr.station, tr.location) for tr in traces):
            for sta in self:
                if sta.nsl == nsl:
                    stations.append(sta)
                    break
            else:
                raise ValueError(f"could not find a station for {nsl}")
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

    def get_coordinates(self, system: CoordSystem = "geographic") -> np.ndarray:
        if self._cached_coordinates is None:
            self._cached_coordinates = np.array(
                [(*sta.effective_lat_lon, sta.effective_elevation) for sta in self]
            )
        return self._cached_coordinates

    @root_validator
    def _load_stations(cls, values) -> Any:  # noqa: N805
        loaded_stations = []
        for path in values.get("pyrocko_station_yamls"):
            loaded_stations += load_stations(filename=str(path))

        for path in values.get("station_xmls"):
            station_xml = load_xml(filename=str(path))
            loaded_stations += station_xml.get_pyrocko_stations()

        for sta in loaded_stations:
            sta = Station.from_pyrocko_station(sta)
            if sta not in values.get("stations"):
                values.get("stations").append(sta)

        # Check stations
        seen_nsls = set()
        for sta in values.get("stations").copy():
            if not sta.lat or not sta.lon:
                logger.warning(
                    "blacklisting station %s: bad geographical coordinates",
                    sta.pretty_nsl,
                )
                values.get("blacklist").add(sta.pretty_nsl)
                continue

            if sta.pretty_nsl in seen_nsls:
                logger.warning("removing doublicate station: %s", sta.pretty_nsl)
                values.get("stations").remove(sta)
                continue
            seen_nsls.add(sta.pretty_nsl)

        if not values.get("stations"):
            raise AttributeError(
                "no stations available, add stations to start detection"
            )

        return values

    def dump_pyrocko_stations(self, filename: Path) -> None:
        dump_stations_yaml(
            [sta.to_pyrocko_station() for sta in self],
            filename=str(filename),
        )

    def __hash__(self) -> int:
        return hash(sta for sta in self)
