from __future__ import annotations

import logging
import re
import zipfile
from datetime import datetime, timedelta
from functools import cached_property
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from lru import LRU
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    constr,
    model_validator,
)
from pyrocko import orthodrome as od
from pyrocko import spit
from pyrocko.cake import LayeredModel, PhaseDef, load_model, m2d
from pyrocko.gf import meta
from rich.progress import Progress

from lassie.octree import get_node_coordinates
from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.utils import (
    CACHE_DIR,
    PhaseDescription,
    datetime_now,
    human_readable_bytes,
    log_call,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from lassie.models.location import Location
    from lassie.models.station import Stations
    from lassie.octree import Node, Octree

logger = logging.getLogger(__name__)

KM = 1e3
GiB = int(1024**3)
MAX_DBS = 16

LRU_CACHE_SIZE = 2000


# TODO: Move to a separate file
DEFAULT_VELOCITY_MODEL = """
-1.00    5.50    3.59    2.7
 0.00    5.50    3.59    2.7
 1.00    5.50    3.59    2.7
 1.00    6.00    3.92    2.7
 4.00    6.00    3.92    2.7
 4.00    6.20    4.05    2.7
 8.00    6.20    4.05    2.7
 8.00    6.30    4.12    2.7
13.00    6.30    4.12    2.7
13.00    6.40    4.18    2.7
17.00    6.40    4.18    2.7
17.00    6.50    4.25    2.7
22.00    6.50    4.25    2.7
22.00    6.60    4.31    2.7
26.00    6.60    4.31    2.7
26.00    6.80    4.44    2.7
30.00    6.80    4.44    2.7
30.00    8.10    5.29    2.7
45.00    8.10    5.29    2.7
"""

DEFAULT_VELOCITY_MODEL_FILE = CACHE_DIR / "velocity_models" / "default.nd"
if not DEFAULT_VELOCITY_MODEL_FILE.exists():
    DEFAULT_VELOCITY_MODEL_FILE.parent.mkdir(exist_ok=True)
    DEFAULT_VELOCITY_MODEL_FILE.write_text(DEFAULT_VELOCITY_MODEL)


class CakeArrival(ModelledArrival):
    tracer: Literal["CakeArrival"] = "CakeArrival"
    phase: str


class EarthModel(BaseModel):
    filename: FilePath | None = Field(
        DEFAULT_VELOCITY_MODEL_FILE,
        description="Path to velocity model.",
    )
    format: Literal["nd", "hyposat"] = Field(
        "nd",
        description="Format of the velocity model. nd or hyposat is supported.",
    )
    crust2_profile: constr(to_upper=True) | tuple[float, float] = Field(
        "",
        description="Crust2 profile name or a tuple of (lat, lon) coordinates.",
    )

    raw_file_data: str | None = Field(None, description="Raw .nd file data.")
    _layered_model: LayeredModel = PrivateAttr()

    model_config = ConfigDict(ignored_types=(cached_property,))

    @model_validator(mode="after")
    def load_model(self) -> EarthModel:
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
                    crust2_profile=self.crust2_profile or None,
                )
        elif self.crust2_profile:
            self._layered_model = load_model(crust2_profile=self.crust2_profile)
        else:
            raise AttributeError("No velocity model or crust2 profile defined.")
        return self

    def trim(self, depth_max: float) -> None:
        """Trim the model to a maximum depth.

        Args:
            depth_max (float): Maximum depth in meters.
        """
        logger.debug("trimming earth model to %.1f km depth", depth_max / KM)
        self._layered_model = self.layered_model.extract(depth_max=depth_max)

    @property
    def layered_model(self) -> LayeredModel:
        return self._layered_model

    def get_profile_vp(self) -> np.ndarray:
        return self.layered_model.profile("vp")

    def get_profile_vs(self) -> np.ndarray:
        return self.layered_model.profile("vs")

    @cached_property
    def hash(self) -> str:
        model_serialised = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            self.layered_model.profile(param).dump(model_serialised)
        return sha1(model_serialised.getvalue()).hexdigest()


class Timing(BaseModel):
    definition: constr(strip_whitespace=True) = "P,p"

    def as_phase_defs(self) -> list[PhaseDef]:
        return [PhaseDef(definition=phase) for phase in self.definition.split(",")]

    def as_pyrocko_timing(self) -> meta.Timing:
        return meta.Timing(f"{{stored:{self.id}}}")

    @property
    def id(self) -> str:
        return re.sub(r"[\,\s\;]", "", self.definition)


class TravelTimeTree(BaseModel):
    earthmodel: EarthModel
    timing: Timing

    distance_bounds: tuple[float, float]
    source_depth_bounds: tuple[float, float]
    receiver_depth_bounds: tuple[float, float]
    time_tolerance: float
    spatial_tolerance: float

    created: datetime = Field(default_factory=datetime_now)

    _sptree: spit.SPTree | None = PrivateAttr(None)
    _file: Path | None = PrivateAttr(None)

    _cached_stations: Stations = PrivateAttr()
    _cached_station_indeces: dict[str, int] = PrivateAttr({})
    _node_lut: dict[bytes, np.ndarray] = PrivateAttr(
        default_factory=lambda: LRU(LRU_CACHE_SIZE)
    )

    def calculate_tree(self) -> spit.SPTree:
        layered_model = self.earthmodel.layered_model

        def evaluate(args) -> float | None:
            receiver_depth, source_depth, distances = args
            rays = layered_model.arrivals(
                phases=self.timing.as_phase_defs(),
                distances=[distances * m2d],
                zstart=source_depth,
                zstop=receiver_depth,
            )
            times = np.fromiter((ray.t for ray in rays), float)
            return times.min() if times.size else None

        spatial_bounds = [
            self.receiver_depth_bounds,
            self.source_depth_bounds,
            self.distance_bounds,
        ]
        return spit.SPTree(
            f=evaluate,
            xbounds=np.array(spatial_bounds),
            ftol=np.array(self.time_tolerance),
            xtols=[
                self.spatial_tolerance,
                self.spatial_tolerance,
                self.spatial_tolerance,
            ],
        )

    def is_suited(
        self,
        timing: Timing,
        earthmodel: EarthModel,
        distance_bounds: tuple[float, float],
        source_depth_bounds: tuple[float, float],
        receiver_depth_bounds: tuple[float, float],
        time_tolerance: float,
        spatial_tolerance: float,
    ) -> bool:
        def check_bounds(self, requested) -> bool:
            return self[0] <= requested[0] and self[1] >= requested[1]

        return (
            str(self.earthmodel.layered_model) == str(earthmodel.layered_model)
            and self.timing == timing
            and check_bounds(self.distance_bounds, distance_bounds)
            and check_bounds(self.source_depth_bounds, source_depth_bounds)
            and check_bounds(self.receiver_depth_bounds, receiver_depth_bounds)
            and self.time_tolerance <= time_tolerance
            and self.spatial_tolerance <= spatial_tolerance
        )

    @property
    def filename(self) -> Path:
        return Path(f"{self.timing.id}-{self.earthmodel.hash}.sptree")

    @classmethod
    def new(cls, **data) -> Self:
        """Create new SPTree and calculate traveltime table.

        Takes all input arguments as the __init__.
        """
        model = cls(**data)
        model._sptree = model.calculate_tree()
        return model

    def save(self, path: Path) -> Path:
        """Save the model and traveltimes to an .sptree archive.

        Args:
            folder (Path): Folder or file to save tree into. If path is a folder a
                native name from the model's hash is used

        Returns:
            Path: Path to the saved archive.
        """
        file = path / self.filename if path.is_dir() else path
        logger.info("saving traveltimes to %s", file)

        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr(
                "model.json",
                self.model_dump_json(
                    indent=2,
                    exclude={"earthmodel": {"nd_file"}},
                ),
            )
            with NamedTemporaryFile() as tmpfile:
                self._get_sptree().dump(tmpfile.name)
                archive.write(tmpfile.name, "model.sptree")
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        """Load model from archive file.

        Args:
            file (Path): Path to archive file.

        Returns:
            Self: Loaded SPTreeModel
        """
        logger.debug("loading traveltimes from %s", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model_file = path / "model.json"
            model = cls.model_validate_json(model_file.read_text())
        model._file = file
        return model

    def _load_sptree(self) -> spit.SPTree:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with zipfile.ZipFile(
            self._file, "r"
        ) as archive, TemporaryDirectory() as temp_dir:
            archive.extract("model.sptree", path=temp_dir)
            return spit.SPTree(filename=str(Path(temp_dir) / "model.sptree"))

    def _get_sptree(self) -> spit.SPTree:
        if self._sptree is None:
            self._sptree = self._load_sptree()
        return self._sptree

    def _interpolate_traveltimes_sptree(
        self,
        coordinates: np.ndarray | list[float],
    ) -> np.ndarray:
        sptree = self._get_sptree()
        timing = self.timing.as_pyrocko_timing()

        coordinates = np.atleast_2d(np.ascontiguousarray(coordinates))
        return timing.evaluate(
            lambda phase: sptree.interpolate_many,
            coordinates,
        )

    def init_lut(self, octree: Octree, stations: Stations) -> None:
        self._cached_stations = stations
        self._cached_station_indeces = {
            sta.pretty_nsl: idx for idx, sta in enumerate(stations)
        }
        station_traveltimes = self.interpolate_travel_times(octree, stations)

        for node, traveltimes in zip(octree, station_traveltimes, strict=True):
            self._node_lut[node.hash()] = traveltimes.astype(np.float32)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling traveltimes LUT for %d nodes", len(nodes))
        stations = self._cached_stations

        node_coords = get_node_coordinates(nodes, system="geographic")
        sta_coords = stations.get_coordinates(system="geographic")

        sta_coords = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T

        receiver_distances = np.linalg.norm(
            sta_coords - node_coords[:, np.newaxis], axis=2
        )

        traveltimes = self._interpolate_travel_times(
            receiver_distances,
            np.array([sta.effective_depth for sta in stations]),
            np.array([node.depth for node in nodes]),
        )

        for node, times in zip(nodes, traveltimes, strict=True):
            self._node_lut[node.hash()] = times.astype(np.float32)

    def lut_fill_level(self) -> float:
        """Return the fill level of the LUT as a float between 0.0 and 1.0"""
        return len(self._node_lut) / self._node_lut.get_size()

    def get_travel_times(self, octree: Octree, stations: Stations) -> np.ndarray:
        station_indices = np.fromiter(
            (self._cached_station_indeces[sta.pretty_nsl] for sta in stations),
            dtype=int,
        )

        stations_traveltimes = []
        fill_nodes = []
        for node in octree:
            try:
                node_traveltimes = self._node_lut[node.hash()][station_indices]
            except KeyError:
                fill_nodes.append(node)
                continue
            stations_traveltimes.append(node_traveltimes)

        if fill_nodes:
            self.fill_lut(fill_nodes)

            cache_hits, cache_misses = self._node_lut.get_stats()
            cache_hit_rate = cache_hits / (cache_hits + cache_misses)
            logger.info(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                self.lut_fill_level() * 100,
                cache_hit_rate * 100,
            )
            return self.get_travel_times(octree, stations)

        return np.asarray(stations_traveltimes).astype(float, copy=False)

    def interpolate_travel_times(
        self,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        receiver_distances = octree.distances_stations(stations)
        receiver_depths = np.array([sta.effective_depth for sta in stations])
        source_depths = np.array([node.depth for node in octree])

        return self._interpolate_travel_times(
            receiver_distances, receiver_depths, source_depths
        )

    def _interpolate_travel_times(
        self,
        receiver_distances: np.ndarray,
        receiver_depths: np.ndarray,
        source_depths: np.ndarray,
    ) -> np.ndarray:
        coordinates = []
        for distances, source_depth in zip(
            receiver_distances,
            source_depths,
            strict=True,
        ):
            node_receivers_distances = (
                receiver_depths,
                np.full_like(distances, source_depth),
                distances,
            )
            coordinates.append(np.asarray(node_receivers_distances).T)

        n_nodes = len(coordinates)
        with Progress() as progress:
            status = progress.add_task(
                f"interpolating station traveltimes for {n_nodes} nodes",
                total=len(coordinates),
            )
            traveltimes = []
            for coords in coordinates:
                traveltimes.append(self._interpolate_traveltimes_sptree(coords))
                progress.update(status, advance=1)

        return np.asarray(traveltimes).astype(float)

    def get_traveltime(self, source: Location, receiver: Location) -> float:
        coordinates = [
            receiver.effective_depth,
            source.effective_depth,
            receiver.distance_to(source),
        ]
        try:
            traveltime = self._get_sptree().interpolate(coordinates) or np.nan
        except spit.OutOfBounds:
            traveltime = np.nan
        return float(traveltime)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[PhaseDescription, Timing] = {
        "cake:P": Timing(definition="P,p"),
        "cake:S": Timing(definition="S,s"),
    }
    earthmodel: EarthModel = EarthModel()
    trim_earth_model_depth: bool = Field(
        True, description="Trim earth model to max depth of the octree."
    )
    lut_cache_size: ByteSize = Field(
        4 * GiB, description="Size of the LUT cache in MB."
    )

    _traveltime_trees: dict[PhaseDescription, TravelTimeTree] = PrivateAttr({})

    @property
    def cache_dir(self) -> Path:
        path = CACHE_DIR / "cake"
        path.mkdir(exist_ok=True)
        return path

    def clear_cache(self) -> None:
        """Clear cached SPTreeModels from user's cache."""
        logging.info("clearing traveltime cached trees in %s", self.cache_dir)
        for file in self.cache_dir.glob("*.sptree"):
            file.unlink()

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.timings.keys())

    def get_vmin(self) -> float:
        earthmodel = self.earthmodel
        vel = np.concatenate((earthmodel.get_profile_vp(), earthmodel.get_profile_vs()))
        return float((vel[vel != 0.0]).min())

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        global LRU_CACHE_SIZE

        bytes_per_node = stations.n_stations * np.float32().itemsize
        n_trees = len(self.timings)
        LRU_CACHE_SIZE = int(self.lut_cache_size / bytes_per_node / n_trees)

        node_cache_fraction = LRU_CACHE_SIZE / octree.maximum_number_nodes()
        logging.info(
            "limiting traveltime LUT size to %d nodes (%s),"
            " caching %.1f%% of possible octree nodes",
            LRU_CACHE_SIZE,
            human_readable_bytes(self.lut_cache_size),
            node_cache_fraction * 100,
        )

        cached_trees = [
            TravelTimeTree.load(file) for file in self.cache_dir.glob("*.sptree")
        ]
        logger.debug("loaded %d cached traveltime trees", len(cached_trees))

        distances = octree.distances_stations(stations)
        source_depths = np.asarray(octree.depth_bounds)
        receiver_depths = np.fromiter((sta.effective_depth for sta in stations), float)

        receiver_depths_bounds = (receiver_depths.min(), receiver_depths.max())
        source_depth_bounds = (source_depths.min(), source_depths.max())
        distance_bounds = (distances.min(), distances.max())
        # FIXME: Time tolerance is too hardcoded. Is 5x a good value?
        time_tolerance = octree.smallest_node_size() / (self.get_vmin() * 5.0)

        # if self.trim_earth_model_depth:
        #     self.earthmodel.trim(-source_depth_bounds[1])

        traveltime_tree_args = {
            "earthmodel": self.earthmodel,
            "distance_bounds": distance_bounds,
            "source_depth_bounds": source_depth_bounds,
            "receiver_depth_bounds": receiver_depths_bounds,
            "spatial_tolerance": octree.size_limit / 2,
            "time_tolerance": time_tolerance,
        }

        for phase_descr, timing in self.timings.items():
            for tree in cached_trees:
                if tree.is_suited(timing=timing, **traveltime_tree_args):
                    logger.info("using cached traveltime tree for %s", phase_descr)
                    break
            else:
                logger.info("pre-calculating traveltime tree for %s", phase_descr)
                tree = TravelTimeTree.new(timing=timing, **traveltime_tree_args)
                tree.save(self.cache_dir)

            tree.init_lut(octree, stations)
            self._traveltime_trees[phase_descr] = tree

    def _get_sptree_model(self, phase: str) -> TravelTimeTree:
        return self._traveltime_trees[phase]

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")
        tree = self._get_sptree_model(phase)
        return tree.get_traveltime(source, receiver)

    @log_call
    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")
        return self._get_sptree_model(phase).get_travel_times(octree, stations)

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[CakeArrival | None]:
        traveltimes = self.get_traveltimes_locations(
            phase,
            source=source,
            receivers=receivers,
        )
        arrivals = []
        for traveltime, _receiver in zip(traveltimes, receivers, strict=True):
            if np.isnan(traveltime):
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = CakeArrival(time=arrivaltime, phase=phase)
            arrivals.append(arrival)
        return arrivals
