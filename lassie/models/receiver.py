from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel
from pyrocko.model import Station as PyrockoStation

from lassie.models.location import Location


class Receiver(Location):
    nsl: tuple[str, str, str]

    def __hash__(self) -> int:
        return super().__hash__() + hash(self.nsl)

    @classmethod
    def from_pyrocko_station(cls, station: PyrockoStation) -> Receiver:
        return cls(
            nsl=station.nsl(),
            lat=station.lat,
            lon=station.lon,
            east_shift=station.east_shift,
            north_shift=station.north_shift,
            elevation=station.elevation,
            depth=station.depth,
        )


class Receivers(BaseModel):
    __root__: list[Receiver] = []

    def __iter__(self) -> Iterator[Receiver]:
        yield from self.__root__

    @classmethod
    def from_pyrocko_stations(cls, stations: list[PyrockoStation]) -> Receivers:
        return Receivers(
            __root__=[Receiver.from_pyrocko_station(sta) for sta in stations]
        )
