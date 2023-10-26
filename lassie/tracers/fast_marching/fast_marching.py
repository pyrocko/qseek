from __future__ import annotations

import asyncio
import functools
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from lru import LRU
from pydantic import BaseModel, ByteSize, Field, PrivateAttr
from pyrocko.modelling import eikonal
from rich.progress import Progress
from scipy.interpolate import RegularGridInterpolator
from typing_extensions import Self

from lassie.models.location import Location
from lassie.models.station import Station, Stations
from lassie.octree import Node
from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    VelocityModel3D,
    VelocityModels,
)
from lassie.utils import CACHE_DIR, PhaseDescription, datetime_now, human_readable_bytes

if TYPE_CHECKING:
    from lassie.octree import Octree


FMM_CACHE_DIR = CACHE_DIR / "fast-marching-cache"

KM = 1e3
GiB = int(1024**3)

logger = logging.getLogger(__name__)


class FastMarchingArrival(ModelledArrival):
    tracer: Literal["FastMarchingArrival"] = "FastMarchingArrival"
    phase: PhaseDescription


class StationTravelTimeVolume(BaseModel):
    center: Location
    station: Station

    velocity_model_hash: str

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]
    grid_spacing: float

    created: datetime = Field(default_factory=datetime_now)

    _travel_times: np.ndarray | None = PrivateAttr(None)

    _north_coords: np.ndarray = PrivateAttr()
    _east_coords: np.ndarray = PrivateAttr()
    _depth_coords: np.ndarray = PrivateAttr()

    # Cached values
    _file: Path | None = PrivateAttr(None)
    _interpolator: RegularGridInterpolator | None = PrivateAttr(None)

    @property
    def travel_times(self) -> np.ndarray:
        if self._travel_times is None:
            self._travel_times = self._load_travel_times()
        return self._travel_times

    def has_travel_times(self) -> bool:
        return self._travel_times is not None or self._file is not None

    def free_cache(self):
        self._interpolator = None
        if self._file is not None:
            logger.warning("cannot free travel time cache, file is not saved")
            self._travel_times = None

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self._east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1],
            grid_spacing,
        )
        self._north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1],
            grid_spacing,
        )
        self._depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1],
            grid_spacing,
        )

    @classmethod
    async def calculate_from_eikonal(
        cls,
        model: VelocityModel3D,
        station: Station,
        save: Path | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> Self:
        arrival_times = model.get_source_arrival_grid(station)

        if not model.is_inside(station):
            offset = station.offset_from(model.center)
            raise ValueError(f"station is outside the velocity model {offset}")

        def eikonal_wrapper(
            velocity_model: VelocityModel3D,
            arrival_times: np.ndarray,
            delta: float,
        ) -> StationTravelTimeVolume:
            logger.debug(
                "calculating travel time volume for %s, grid size %s, spacing %s m...",
                station.pretty_nsl,
                arrival_times.shape,
                velocity_model.grid_spacing,
            )
            eikonal.eikonal_solver_fmm_cartesian(
                velocity_model._velocity_model,
                arrival_times,
                delta=delta,
            )
            station_travel_times = cls(
                center=model.center,
                velocity_model_hash=model.hash(),
                station=station,
                east_bounds=model.east_bounds,
                north_bounds=model.north_bounds,
                depth_bounds=model.depth_bounds,
                grid_spacing=model.grid_spacing,
            )
            station_travel_times._travel_times = arrival_times.astype(np.float32)
            if save:
                station_travel_times.save(save)

            return station_travel_times

        loop = asyncio.get_running_loop()
        work = functools.partial(
            eikonal_wrapper,
            model,
            arrival_times,
            delta=model.grid_spacing,
        )

        return await loop.run_in_executor(executor, work)

    @property
    def filename(self) -> str:
        # TODO: Add origin to hash to avoid collisions
        return f"{self.station.pretty_nsl}-{self.velocity_model_hash}.3dtt"

    def get_travel_time_interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            self._interpolator = RegularGridInterpolator(
                (self._east_coords, self._north_coords, self._depth_coords),
                self.travel_times,
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def interpolate_travel_time(
        self,
        location: Location,
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> float:
        interpolator = self.get_travel_time_interpolator()
        offset = location.offset_from(self.center)
        return interpolator([offset], method=method).astype(float, copy=False)[0]

    def interpolate_nodes(
        self,
        nodes: Sequence[Node],
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> np.ndarray:
        interpolator = self.get_travel_time_interpolator()

        coordinates = [node.as_location().offset_from(self.center) for node in nodes]
        return interpolator(coordinates, method=method).astype(float, copy=False)

    def get_meshgrid(self) -> list[np.ndarray]:
        return np.meshgrid(
            self._east_coords,
            self._north_coords,
            self._depth_coords,
            indexing="ij",
        )

    def save(self, path: Path) -> Path:
        """Save travel times to a zip file.

        The zip file contains a model.json file with the model metadata and a
        numpy file with the travel times.

        Args:
            path (Path): path to save the travel times to

        Returns:
            Path: path to the saved travel times
        """
        if not self.has_travel_times():
            raise AttributeError("travel times have not been calculated yet")

        file = path / self.filename if path.is_dir() else path
        logger.debug("saving travel times to %s...", file)

        with zipfile.ZipFile(str(file), "w") as archive:
            archive.writestr("model.json", self.model_dump_json(indent=2))
            travel_times = archive.open("travel_times.npy", "w")
            np.save(travel_times, self.travel_times)
            travel_times.close()

        self._file = file
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        """Load 3D travel times from a .3dtt file.

        Args:
            file (Path): path to the .3dtt file containing the travel times

        Returns:
            Self: 3D travel times
        """
        logger.debug("loading travel times from %s...", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model_file = path / "model.json"
            model = cls.model_validate_json(model_file.read_text())
        model._file = file
        return model

    def _load_travel_times(self) -> np.ndarray:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with zipfile.ZipFile(self._file, "r") as archive:
            return np.load(archive.open("travel_times.npy", "r"))


class FastMarchingTracer(RayTracer):
    tracer: Literal["FastMarchingRayTracer"] = "FastMarchingRayTracer"

    phase: PhaseDescription = "fm:P"
    interpolation_method: Literal["nearest", "linear", "cubic"] = "linear"
    nthreads: int = Field(
        default=0,
        description="Number of threads to use for travel time."
        "If set to 0, cpu_count*2 will be used.",
    )

    lut_cache_size: ByteSize = Field(
        default=2 * GiB,
        description="Size of the LUT cache.",
    )

    velocity_model: VelocityModels = Constant3DVelocityModel()

    _travel_time_volumes: dict[str, StationTravelTimeVolume] = PrivateAttr({})
    _velocity_model: VelocityModel3D | None = PrivateAttr(None)

    _cached_stations: Stations = PrivateAttr()
    _cached_station_indeces: dict[str, int] = PrivateAttr({})
    _node_lut: dict[bytes, np.ndarray] = PrivateAttr()

    def get_available_phases(self) -> tuple[str, ...]:
        return (self.phase,)

    def get_travel_time_volume(self, location: Location) -> StationTravelTimeVolume:
        return self._travel_time_volumes[location.location_hash()]

    def add_travel_time_volume(
        self,
        location: Location,
        volume: StationTravelTimeVolume,
    ) -> None:
        self._travel_time_volumes[location.location_hash()] = volume

    def has_travel_time_volume(self, location: Location) -> bool:
        return location.location_hash() in self._travel_time_volumes

    def lut_fill_level(self) -> float:
        """Return the fill level of the LUT as a float between 0.0 and 1.0"""
        return len(self._node_lut) / self._node_lut.get_size()

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
    ) -> None:
        logger.info("loading velocity model %s", self.velocity_model.model)
        velocity_model = self.velocity_model.get_model(octree)

        logger.info(
            "3D velocity model dimensions: "
            " north-south %.1f m, east-west %.1f m , depth %.1f m, grid spacing %s m",
            velocity_model.north_size,
            velocity_model.east_size,
            velocity_model.depth_size,
            velocity_model.grid_spacing,
        )
        logger.info("3D velocity model reference location %s", velocity_model.center)
        self._velocity_model = velocity_model

        for station in stations:
            if not velocity_model.is_inside(station):
                offset = station.offset_from(velocity_model.center)
                stations.blacklist_station(
                    station,
                    reason=f"outside fast-marching velocity model, offset {offset}",
                )

        for station in stations:
            velocity_station = velocity_model.get_velocity(station)
            if velocity_station <= 0.0:
                raise ValueError(
                    f"station {station.pretty_nsl} has negative or zero velocity"
                    f" {velocity_station} m/s"
                )
            logger.info(
                "velocity at station %s: %.1f m/s",
                station.pretty_nsl,
                velocity_station,
            )

        nodes_covered = [
            node for node in octree if velocity_model.is_inside(node.as_location())
        ]
        if not nodes_covered:
            raise ValueError("no octree node is inside the velocity model")

        logger.info(
            "%d%% octree nodes are inside the %s velocity model",
            len(nodes_covered) / octree.n_nodes * 100,
            self.phase,
        )

        self._cached_stations = stations
        self._cached_station_indeces = {
            sta.pretty_nsl: idx for idx, sta in enumerate(stations)
        }
        bytes_per_node = stations.n_stations * np.float32().itemsize
        lru_cache_size = int(self.lut_cache_size / bytes_per_node)
        self._node_lut = LRU(lru_cache_size)

        # TODO: This should be total number nodes. Not only leaf nodes.
        node_cache_fraction = lru_cache_size / octree.total_number_nodes()
        logging.info(
            "limiting traveltime LUT size to %d nodes (%s),"
            " caching %.1f%% of possible octree nodes",
            lru_cache_size,
            human_readable_bytes(self.lut_cache_size),
            node_cache_fraction * 100,
        )

        cache_dir = FMM_CACHE_DIR / f"{velocity_model.hash()}"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        else:
            self._load_cached_tavel_times(cache_dir)

        calc_stations = [
            station for station in stations if not self.has_travel_time_volume(station)
        ]
        await self._calculate_travel_times(calc_stations, cache_dir)

    def _load_cached_tavel_times(self, cache_dir: Path) -> None:
        logger.debug("loading travel times volumes from cache %s...", cache_dir)
        volumes: dict[str, StationTravelTimeVolume] = {}
        for file in cache_dir.glob("*.3dtt"):
            try:
                travel_times = StationTravelTimeVolume.load(file)
            except zipfile.BadZipFile:
                logger.warning("removing bad travel time file %s", file)
                file.unlink()
                continue
            volumes[travel_times.station.location_hash()] = travel_times

        logger.info(
            "loaded %d travel times volumes for %s from cache", len(volumes), self.phase
        )
        self._travel_time_volumes.update(volumes)

    async def _calculate_travel_times(
        self,
        stations: list[Station],
        cache_dir: Path,
    ) -> None:
        nthreads = self.nthreads if self.nthreads > 0 else os.cpu_count() * 2
        executor = ThreadPoolExecutor(
            max_workers=nthreads,
            thread_name_prefix="lassie-fmm",
        )
        if self._velocity_model is None:
            raise AttributeError("velocity model has not been prepared yet")

        async def worker_station_travel_time(station: Station) -> None:
            volume = await StationTravelTimeVolume.calculate_from_eikonal(
                self._velocity_model,
                station,
                save=cache_dir,
                executor=executor,
            )
            self.add_travel_time_volume(station, volume)

        calculate_work = [worker_station_travel_time(station) for station in stations]
        if not calculate_work:
            return

        start = datetime_now()
        tasks = [asyncio.create_task(work) for work in calculate_work]
        with Progress() as progress:
            status = progress.add_task(
                f"calculating travel time volumes for {len(tasks)} stations"
                f" ({nthreads} threads)",
                total=len(tasks),
            )
            for _task in asyncio.as_completed(tasks):
                await _task
                progress.advance(status)
        logger.info("calculated travel time volumes in %s", datetime_now() - start)

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        if phase != self.phase:
            raise ValueError(f"phase {phase} is not supported by this tracer")

        volume = self.get_travel_time_volume(receiver)
        return volume.interpolate_travel_time(
            source,
            method=self.interpolation_method,
        )

    def get_travel_times(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        if phase != self.phase:
            raise ValueError(f"phase {phase} is not supported by this tracer")

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
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                self.lut_fill_level() * 100,
                cache_hit_rate * 100,
            )
            return self.get_travel_times(phase, octree, stations)

        return np.asarray(stations_traveltimes).astype(float, copy=False)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        travel_times = []
        n_nodes = len(nodes)

        with Progress() as progress:
            status = progress.add_task(
                f"interpolating {self.phase} travel times for {n_nodes} nodes",
                total=self._cached_stations.n_stations,
            )
            for station in self._cached_stations:
                volume = self.get_travel_time_volume(station)
                travel_times.append(volume.interpolate_nodes(nodes))
                progress.advance(status)

        travel_times = np.array(travel_times).T

        for node, station_travel_times in zip(nodes, travel_times, strict=True):
            self._node_lut[node.hash()] = station_travel_times.copy()

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
        if phase != self.phase:
            raise ValueError(f"phase {phase} is not supported by this tracer")

        traveltimes = []
        for receiver in receivers:
            traveltimes.append(self.get_travel_time_location(phase, source, receiver))

        arrivals = []
        for traveltime, _receiver in zip(traveltimes, receivers, strict=True):
            if np.isnan(traveltime):
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = FastMarchingArrival(time=arrivaltime, phase=phase)
            arrivals.append(arrival)
        return arrivals
