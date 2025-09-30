from __future__ import annotations

import logging
import re
import struct
import zipfile
from datetime import datetime, timedelta
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Literal, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
)
from pyrocko import spit
from pyrocko.cake import PhaseDef, m2d
from pyrocko.gf import meta

from qseek.cache_lru import ArrayLRUCache
from qseek.stats import get_progress
from qseek.tracers.base import ModelledArrival, RayTracer
from qseek.tracers.utils import EarthModel, surface_distances
from qseek.utils import (
    CACHE_DIR,
    PhaseDescription,
    alog_call,
    datetime_now,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from qseek.models.location import Location
    from qseek.models.station import Stations
    from qseek.octree import Node, Octree

logger = logging.getLogger(__name__)

KM = 1e3


class Timing(BaseModel):
    definition: Annotated[str, StringConstraints(strip_whitespace=True)] = "P,p"

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
    _cached_station_indices: dict[str, int] = PrivateAttr({})
    _node_lut: ArrayLRUCache[bytes] = PrivateAttr()

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
        def check_bounds(self, requested: tuple[float, float]) -> bool:
            return self[0] <= requested[0] and self[1] >= requested[1]

        return (
            self.earthmodel.hash == earthmodel.hash
            and self.timing == timing
            and check_bounds(self.distance_bounds, distance_bounds)
            and check_bounds(self.source_depth_bounds, source_depth_bounds)
            and check_bounds(self.receiver_depth_bounds, receiver_depth_bounds)
            and self.time_tolerance <= time_tolerance
            and self.spatial_tolerance <= spatial_tolerance
        )

    @property
    def filename(self) -> Path:
        hash = sha1(self.earthmodel.hash.encode())
        hash.update(
            struct.pack(
                "dddddddd",
                *self.distance_bounds,
                *self.source_depth_bounds,
                *self.receiver_depth_bounds,
                self.time_tolerance,
                self.spatial_tolerance,
            )
        )
        return Path(f"{self.timing.id}-{hash.hexdigest()}.sptree")

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
            path (Path): Folder or file to save tree into. If path is a folder a
                native name from the model's hash is used

        Returns:
            Path: Path to the saved archive.
        """
        file = path / self.filename if path.is_dir() else path
        logger.info("saving traveltimes to %s", file)
        self.earthmodel.fortify()

        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr(
                "model.json",
                self.model_dump_json(
                    indent=2,
                    exclude={"earthmodel": {"filename"}},
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
        logger.debug("loading cached traveltimes from %s", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model_file = path / "model.json"
            model = cls.model_validate_json(model_file.read_text())
        model._file = file
        return model

    def _get_sptree(self) -> spit.SPTree:
        if self._sptree is None:
            if not self._file or not self._file.exists():
                raise FileNotFoundError(f"file {self._file} not found")

            with (
                zipfile.ZipFile(self._file, "r") as archive,
                TemporaryDirectory() as temp_dir,
            ):
                archive.extract("model.sptree", path=temp_dir)
                self._sptree = spit.SPTree(
                    filename=str(Path(temp_dir) / "model.sptree")
                )
        return self._sptree

    async def init_lut(self, nodes: Sequence[Node], stations: Stations) -> None:
        self._node_lut = ArrayLRUCache(
            f"cake-ttt-{self.timing.id}",
            short_name=self.timing.id,
        )
        self._cached_stations = stations
        self._cached_station_indices = {
            sta.nsl.pretty: idx for idx, sta in enumerate(stations)
        }
        logger.info(
            "warming up traveltime LUT %s for %d stations and %d nodes",
            self.timing.definition,
            stations.n_stations,
            len(nodes),
        )
        await self.fill_lut(nodes)

    async def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling traveltimes LUT for %d nodes", len(nodes))
        stations = self._cached_stations

        traveltimes = await self._interpolate_travel_times(
            surface_distances(nodes, stations),
            np.array([sta.effective_depth for sta in stations]),
            np.array([node.as_location().effective_depth for node in nodes]),
        )

        node_lut = self._node_lut
        for node, times in zip(nodes, traveltimes, strict=True):
            if np.all(np.isnan(times)):
                logger.warning("no traveltimes for node %s", node)
            node_lut[node.hash()] = times.astype(np.float32)

    async def get_travel_times(
        self,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        try:
            station_indices = np.fromiter(
                (self._cached_station_indices[sta.nsl.pretty] for sta in stations),
                dtype=int,
            )
        except KeyError as exc:
            raise ValueError(
                "stations not found in cached stations, "
                "was the LUT initialized with `TravelTimeTree.init_lut`?"
            ) from exc

        stations_travel_times = []
        fill_nodes = []
        node_lut = self._node_lut
        for node in nodes:
            try:
                node_travel_times = node_lut[node.hash()][station_indices]
            except KeyError:
                fill_nodes.append(node)
                continue
            stations_travel_times.append(node_travel_times)

        if fill_nodes:
            await self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_travel_times(nodes, stations)

        return np.asarray(stations_travel_times).astype(float, copy=False)

    async def interpolate_travel_times(
        self,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        receiver_distances = surface_distances(octree.nodes, stations)
        receiver_depths = np.array([sta.effective_depth for sta in stations])
        source_depths = np.array(
            [node.as_location().effective_depth for node in octree]
        )
        return await self._interpolate_travel_times(
            receiver_distances, receiver_depths, source_depths
        )

    async def _interpolate_travel_times(
        self,
        receiver_distances: np.ndarray,
        receiver_depths: np.ndarray,
        source_depths: np.ndarray,
    ) -> np.ndarray:
        sptree = self._get_sptree()
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
            coordinates.append(np.asarray(node_receivers_distances).T.copy())

        n_nodes = len(coordinates)
        travel_times = []
        with get_progress() as progress:
            status = progress.add_task(
                f"interpolating {self.timing.definition} travel times"
                f" for {n_nodes} nodes",
                total=len(coordinates),
            )
            for coords in coordinates:
                travel_times.append(sptree.interpolate_many(np.atleast_2d(coords)))
                progress.update(status, advance=1)

            progress.remove_task(status)

        return np.asarray(travel_times).astype(float)

    def get_travel_time(self, source: Location, receiver: Location) -> float:
        coordinates = [
            receiver.effective_depth,
            source.effective_depth,
            receiver.surface_distance_to(source),
        ]
        try:
            travel_time = self._get_sptree().interpolate(coordinates) or np.nan
        except spit.OutOfBounds:
            travel_time = np.nan
        return float(travel_time)


class CakeTracer(RayTracer):
    """Travel time calculation for 1D layered velocity models.

    Calculation is based on Pyrocko Cake ray tracer.
    """

    tracer: Literal["CakeTracer"] = "CakeTracer"
    phases: dict[PhaseDescription, Timing] = Field(
        default={
            "cake:P": Timing(definition="P,p"),
            "cake:S": Timing(definition="S,s"),
        },
        description="Dictionary of phases and timings to calculate.",
    )
    earthmodel: EarthModel = Field(
        default_factory=EarthModel,
        description="Earth model to calculate travel times for.",
    )

    _travel_time_trees: dict[PhaseDescription, TravelTimeTree] = PrivateAttr({})

    @property
    def cache_dir(self) -> Path:
        path = CACHE_DIR / "cake"
        path.mkdir(exist_ok=True)
        return path

    def clear_cache(self) -> None:
        """Clear cached SPTreeModels from user's cache."""
        logging.info("clearing cached travel time trees in %s", self.cache_dir)
        for file in self.cache_dir.glob("*.sptree"):
            file.unlink()

    def get_available_phases(self) -> tuple[str, ...]:
        return tuple(self.phases.keys())

    def get_min_velocity_at_depth(self, depth: float) -> float:
        earthmodel = self.earthmodel
        depths = earthmodel.get_profile_depth()

        vp_interpolated = np.interp(depth, depths, earthmodel.get_profile_vp())
        vs_interpolated = np.interp(depth, depths, earthmodel.get_profile_vs())
        return min(vp_interpolated, vs_interpolated)

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        rundir: Path | None = None,
    ) -> None:
        cached_trees = self._load_cached_trees()
        octree.reset()

        distances = surface_distances(octree.leaf_nodes, stations)
        source_depths = (
            np.asarray(octree.depth_bounds) - octree.location.effective_elevation
        )
        receiver_depths = np.array([sta.effective_depth for sta in stations])

        distance_bounds = (distances.min(), distances.max())
        source_depth_bounds = (source_depths.min(), source_depths.max())

        receiver_depth_padding = np.ptp(receiver_depths) * 0.01  # 10% margin
        receiver_depths_bounds = (
            receiver_depths.min() - receiver_depth_padding,
            receiver_depths.max() + receiver_depth_padding,
        )
        # FIXME: Time tolerance is too hardcoded. Is 2x a good value?
        receiver_vmin = np.array(
            [self.get_min_velocity_at_depth(sta.effective_depth) for sta in stations]
        )
        time_tolerance = octree.smallest_node_size() / (receiver_vmin.min() * 4)
        spatial_tolerance = octree.smallest_node_size()

        traveltime_tree_args = {
            "earthmodel": self.earthmodel,
            "distance_bounds": distance_bounds,
            "source_depth_bounds": source_depth_bounds,
            "receiver_depth_bounds": receiver_depths_bounds,
            "spatial_tolerance": spatial_tolerance,
            "time_tolerance": time_tolerance,
        }

        for phase_descr, timing in self.phases.items():
            for tree in cached_trees:
                if tree.is_suited(timing=timing, **traveltime_tree_args):
                    logger.info("using cached travel time tree for %s", phase_descr)
                    logger.debug("from file %s", tree.filename)
                    break
            else:
                logger.info(
                    "pre-calculating travel time tree for %s"
                    " with tolerances %.4f s and %.2f m",
                    phase_descr,
                    time_tolerance,
                    spatial_tolerance,
                )

                tree = TravelTimeTree.new(timing=timing, **traveltime_tree_args)
                tree.save(self.cache_dir)

            await tree.init_lut(octree.leaf_nodes, stations)
            self._travel_time_trees[phase_descr] = tree

        if rundir:
            cake_plots = rundir / "cake"
            cake_plots.mkdir(exist_ok=True)
            for phase, tree in self._travel_time_trees.items():
                tree.earthmodel.save_plot(
                    cake_plots / f"earthmodel_{phase.replace(':', '_')}.png",
                )

    def _get_sptree_model(self, phase: str) -> TravelTimeTree:
        return self._travel_time_trees[phase]

    def _load_cached_trees(self) -> list[TravelTimeTree]:
        trees = []
        for file in self.cache_dir.glob("*.sptree"):
            try:
                tree = TravelTimeTree.load(file)
            except ValidationError:
                logger.warning("deleting invalid cached travel time tree %s", file)
                file.unlink()
                continue
            trees.append(tree)
        logger.debug("loaded %d cached travel time trees", len(trees))
        return trees

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        try:
            tree = self._get_sptree_model(phase)
        except KeyError as exc:
            raise ValueError(f"Phase {phase} is not defined.") from exc
        return tree.get_travel_time(source, receiver)

    @alog_call
    async def get_travel_times(
        self,
        phase: str,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        try:
            return await self._get_sptree_model(phase).get_travel_times(
                nodes,
                stations,
            )
        except KeyError as exc:
            raise ValueError(f"Phase {phase} is not defined.") from exc

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
        traveltimes = self.get_travel_times_locations(
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
            arrival = ModelledArrival(
                time=arrivaltime,
                phase=phase,
            )
            arrivals.append(arrival)
        return arrivals
