from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from pydantic import BaseModel, constr, root_validator
from pyrocko.io.stationxml import load_xml
from pyrocko.model import load_stations

if TYPE_CHECKING:
    from pyrocko.model import Station as PyrockoStation

from lassie.models.location import Location

NSL_RE = r"^[a-zA-Z0-9]{0,2}\.[a-zA-Z0-9]{0,5}\.[a-zA-Z0-9]{0,3}$"


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
        for station in self.stations:
            if ".".join(station.nsl) in self.blacklist:
                continue
            yield station

    def all_nsl(self) -> tuple[tuple[str, str, str]]:
        return tuple(recv.nsl for recv in self)

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
        return values
