from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    GetCoreSchemaHandler,
    PositiveFloat,
)
from pydantic_core import CoreSchema, core_schema
from pyrocko.io.stationxml import load_xml
from pyrocko.model import Station as PyrockoStation
from pyrocko.model import dump_stations_yaml, load_stations

from qseek.utils import _NSL, NSL

if TYPE_CHECKING:
    from pyrocko.trace import Trace


from qseek.models.location import CoordSystem, Location

logger = logging.getLogger(__name__)


class Blacklist(set[NSL]):
    def __contains__(self, other: NSL) -> bool:
        return any(nsl.match(other) for nsl in self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(set[NSL]))


class Station(Location):
    network: str = Field(
        ...,
        description="Seismic network code",
        max_length=2,
    )
    station: str = Field(
        ...,
        description="Station code",
        max_length=5,
    )
    location: str = Field(
        default="",
        description="Location code",
        max_length=2,
    )

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

    def as_pyrocko_station(self) -> PyrockoStation:
        return PyrockoStation(
            **self.model_dump(
                include={
                    "network",
                    "station",
                    "location",
                    "lat",
                    "lon",
                    "north_shift",
                    "east_shift",
                    "depth",
                    "elevation",
                }
            )
        )

    @property
    def nsl(self) -> _NSL:
        """Network Station Location code as tuple.

        Returns:
            tuple[str, str, str]: Network, Station, Location
        """
        return _NSL(self.network, self.station, self.location)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.nsl))


class StationInventory(BaseModel):
    pyrocko_station_yamls: list[FilePath] = Field(
        default=[],
        description="List of [Pyrocko station YAML]"
        "(https://pyrocko.org/docs/current/formats/yaml.html) files.",
    )
    station_xmls: list[FilePath | DirectoryPath] = Field(
        default=[],
        description="List of StationXML files or "
        "directories containing StationXML (.xml) files.",
    )

    blacklist: Blacklist = Field(
        default=Blacklist(),
        description="Blacklist stations and exclude from detecion. "
        "Format is `['NET.STA.LOC', ...]`.",
    )
    stations: list[Station] = []

    max_distance: PositiveFloat | None = Field(
        default=None,
        description="Maximum distance in meters from the centroid location to "
        "include stations for detection. If None, all stations are included.",
    )

    def model_post_init(self, __context: Any) -> None:
        loaded_stations = []
        for file in self.pyrocko_station_yamls:
            loaded_stations += load_stations(filename=str(file.expanduser()))

        for path in self.station_xmls:
            if path.is_dir():
                station_xmls = list(path.glob("*.[xX][mM][lL]"))
            elif path.is_file():
                station_xmls = [path]
            else:
                continue
            for file in station_xmls:
                try:
                    station_xml = load_xml(filename=str(file.expanduser()))
                except StopIteration:
                    logger.error("could not load StationXML file: %s", file)
                    continue
                loaded_stations += station_xml.get_pyrocko_stations()

        for sta in loaded_stations:
            sta = Station.from_pyrocko_station(sta)
            self.stations.append(sta)

        self.sanitize_stations()

    def __iter__(self) -> Iterator[Station]:
        return (sta for sta in self.stations if sta.nsl not in self.blacklist)

    def __contains__(self, other: NSL) -> bool:
        return any(sta.nsl == other for sta in self)

    def sanitize_stations(self) -> None:
        """Remove stations with bad coordinates or duplicates."""
        logger.debug("weeding bad stations")

        seen_nsls = set()
        for sta in self.stations.copy():
            if sta.lat == 0.0 or sta.lon == 0.0:
                logger.warning(
                    "removing station %s with bad coordinates: lat %.4f, lon %.4f",
                    sta.nsl.pretty,
                    sta.lat,
                    sta.lon,
                )
                self.stations.remove(sta)
                continue

            if sta.nsl.pretty in seen_nsls:
                logger.warning("removing duplicate station: %s", sta.nsl.pretty)
                self.stations.remove(sta)
                continue
            seen_nsls.add(sta.nsl.pretty)

        # if not self.stations:
        #     logger.warning("no stations available, add stations to start detection")

    def blacklist_station(self, station: Station, reason: str) -> None:
        logger.warning("blacklisting station %s: %s", station.nsl.pretty, reason)
        self.blacklist.add(station.nsl)
        if self.n_stations == 0:
            raise ValueError("no stations available, all stations blacklisted")

    def filter_stations(self, nsls: Iterable[NSL]) -> None:
        """Filter stations by NSL codes.

        All stations not in the provided NSL codes are removed from the inventory.

        Args:
            nsls (list[NSL]): List of NSL codes to keep.
        """
        n_removed_stations = 0
        for sta in sorted(self.stations, key=lambda s: s.nsl.pretty):
            if sta.nsl not in nsls:
                logger.warning(
                    "removing station %s from inventory",
                    sta.nsl.pretty,
                )
                self.stations.remove(sta)
                n_removed_stations += 1

        if n_removed_stations:
            logger.warning(
                "removed %d stations from inventory",
                n_removed_stations,
            )
        if not self.stations:
            raise ValueError("no stations available, add waveforms to start detection")

    def prepare(self, location: Location) -> None:
        """Prepare the station inventory.

        Args:
            location (Location): Location to calculate distances from.

        Raises:
            ValueError: If no stations are available after preparation.
        """
        logger.info("preparing station inventory")

        if self.max_distance is not None:
            for sta in self.stations.copy():
                distance = location.distance_to(sta)

                if distance > self.max_distance:
                    logger.warning(
                        "removing station %s: distance to octree is %g m",
                        sta.nsl.pretty,
                        distance,
                    )
                    self.stations.remove(sta)

        if not self.stations:
            raise ValueError("no stations available, add stations to start detection")

    @property
    def n_stations(self) -> int:
        """Number of stations."""
        return len(list(self))

    @property
    def n_networks(self) -> int:
        """Number of stations."""
        return len({sta.network for sta in self})

    def get_nsls(self) -> tuple[NSL, ...]:
        """Get all NSL codes from all stations."""
        return tuple(sta.nsl for sta in self)

    def select_from_traces(self, traces: Sequence[Trace]) -> list[Station]:
        """Select stations by NSL code.

        Stations are not unique and are ordered by the input traces.

        Args:
            traces (Iterable[Trace]): Iterable of Pyrocko Traces

        Returns:
            Stations: Containing only selected stations.
        """
        available_stations = {sta.nsl: sta for sta in self}
        try:
            selected_stations = [
                available_stations[_NSL(tr.network, tr.station, tr.location)]
                for tr in traces
            ]
        except KeyError as exc:
            raise ValueError("could not find station information") from exc

        return selected_stations

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
        if system != "geographic":
            raise NotImplementedError("only geographic coordinates are implemented.")
        return np.array(
            [(*sta.effective_lat_lon, sta.effective_elevation) for sta in self]
        )

    def as_pyrocko_stations(self) -> list[PyrockoStation]:
        """Convert the stations to PyrockoStation objects.

        Returns:
            A list of PyrockoStation objects.
        """
        return [sta.as_pyrocko_station() for sta in self]

    def export_pyrocko_stations(self, filename: Path) -> None:
        """Dump stations to pyrocko station yaml file.

        Args:
            filename (Path): Path to yaml file.
        """
        dump_stations_yaml(
            self.as_pyrocko_stations(),
            filename=str(filename.expanduser()),
        )

    def export_csv(self, filename: Path) -> None:
        """Dump stations to CSV file.

        Args:
            filename (Path): Path to CSV file.
        """
        with filename.open("w") as f:
            f.write(
                "network,station,location,latitude,longitude,elevation,depth,WKT_geom\n"
            )
            for sta in self:
                f.write(
                    f"{sta.network},{sta.station},{sta.location},"
                    f"{sta.effective_lat},{sta.effective_lon},{sta.elevation},"
                    f"{sta.depth},{sta.as_wkt()}\n"
                )

    def export_vtk(self, reference: Location | None = None) -> None: ...

    def __hash__(self) -> int:
        return hash(sta for sta in self)


class StationList(Sequence[Station]):
    """A frozen station list."""

    _stations: list[Station]
    _stations: list[Station]
    _indices: dict[str, int]

    def __init__(self, stations: Sequence[Station]) -> None:
        self._stations = list(stations)
        self._indices = {sta.nsl.pretty: idx for idx, sta in enumerate(stations)}

    def __iter__(self) -> Iterator[Station]:
        yield from self._stations

    def __getitem__(self, index: int) -> Station:
        return self._stations[index]

    def __len__(self) -> int:
        return len(self._stations)

    def get_nsls(self) -> tuple[NSL, ...]:
        return tuple(sta.nsl for sta in self._stations)

    def get_index(self, nsl: NSL) -> int:
        """Get the index of a station by its NSL code.

        Args:
            nsl (NSL): NSL code of the station.

        Returns:
            int: Index of the station.
        """
        return self._indices[nsl.pretty]

    def get_indexes(self, stations: Sequence[Station]) -> np.ndarray:
        """Get the indices of stations by their NSL codes.

        Args:
            stations (Iterable[Station]): Iterable of stations.

        Returns:
            np.ndarray: Indices of the stations.
        """
        try:
            return np.fromiter(
                (self._indices[sta.nsl.pretty] for sta in stations), dtype=int
            )
        except KeyError as exc:
            raise ValueError("could not find station information") from exc

    def get_indices_by_nsl(self, nsls: Sequence[NSL]) -> np.ndarray:
        """Get the indices of stations by their NSL codes.

        Args:
            nsls (Iterable[NSL]): Iterable of NSL codes of the stations.

        Returns:
            np.ndarray: Indices of the stations.
        """
        try:
            return np.fromiter((self._indices[nsl.pretty] for nsl in nsls), dtype=int)
        except KeyError as exc:
            raise ValueError("could not find station information") from exc

    @classmethod
    def from_inventory(cls, inventory: StationInventory) -> StationList:
        return cls(stations=list(inventory))

    @property
    def n_stations(self) -> int:
        return len(self._stations)
