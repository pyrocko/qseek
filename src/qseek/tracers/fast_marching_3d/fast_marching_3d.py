from __future__ import annotations

import asyncio
import functools
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, Sequence

import eikonalfm
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pyevtk.hl import gridToVTK
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from qseek.cache_lru import ArrayLRUCache
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory, StationList
from qseek.stats import get_progress
from qseek.tracers.base import ModelledArrival, RayTracer
from qseek.utils import (
    CACHE_DIR,
    PhaseDescription,
    RangeType,
    alog_call,
    datetime_now,
    get_cpu_count,
)

from .velocity_models import (
    Constant3DVelocityModel,
    VelocityModel3D,
    VelocityModels,
    grid_offset,
    node_grid_offsets,
)

if TYPE_CHECKING:
    from qseek.octree import Node, Octree


FMM_CACHE_DIR = CACHE_DIR / "fast-marching-cache"
KM = 1e3

InterpolationMethod = Literal["nearest", "linear", "cubic"]

# Batches of nodes from which the LUT fill shows a progress bar
PROGRESS_MIN_NODES = 50_000
# Cells of odd reflection around a volume for its cubic spline
SPLINE_PAD = 4

logger = logging.getLogger(__name__)


FMMImplementation = Literal["eikonalfm"]


class StationTravelTimeVolume(BaseModel):
    center: Location
    station: Station

    velocity_model_hash: str

    east_bounds: RangeType
    north_bounds: RangeType
    depth_bounds: RangeType
    grid_spacing: float

    fmm_tracer: FMMImplementation

    created: datetime = Field(default_factory=datetime_now)

    _travel_times: np.ndarray | None = PrivateAttr(None)

    _north_coords: np.ndarray = PrivateAttr()
    _east_coords: np.ndarray = PrivateAttr()
    _depth_coords: np.ndarray = PrivateAttr()

    # Cached values
    _file: Path | None = PrivateAttr(None)
    _interpolator: RegularGridInterpolator | None = PrivateAttr(None)
    _spline_coefficients: np.ndarray | None = PrivateAttr(None)

    @property
    def travel_times(self) -> np.ndarray:
        if self._travel_times is None:
            self._travel_times = self._load_travel_times()
        return self._travel_times

    def has_travel_times(self) -> bool:
        return self._travel_times is not None or self._file is not None

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing
        neast = round(self.east_bounds.width() / grid_spacing) + 1
        nnorth = round(self.north_bounds.width() / grid_spacing) + 1
        ndepth = round(self.depth_bounds.width() / grid_spacing) + 1

        self._east_coords = np.linspace(
            self.east_bounds.start,
            self.east_bounds.end,
            neast,
            endpoint=True,
        )
        self._north_coords = np.linspace(
            self.north_bounds.start,
            self.north_bounds.end,
            nnorth,
            endpoint=True,
        )
        self._depth_coords = np.linspace(
            self.depth_bounds.start,
            self.depth_bounds.end,
            ndepth,
            endpoint=True,
        )

    @classmethod
    async def calculate_from_eikonal(
        cls,
        model: VelocityModel3D,
        station: Station,
        save: Path | None = None,
        implementation: FMMImplementation = "eikonalfm",
        executor: ThreadPoolExecutor | None = None,
    ) -> Self:
        if not model.is_inside(station):
            raise ValueError(
                f"station {station.nsl.pretty} is outside the velocity model"
            )

        def eikonal_wrapper(model: VelocityModel3D) -> Self:
            logger.info(
                "calculating travel time volume for %s, spacing %s m using %s...",
                station.nsl.pretty,
                model.grid_spacing,
                implementation,
            )
            if implementation != "eikonalfm":
                raise ValueError(f"unknown eikonal solver {implementation}")
            eikonal_factors = eikonalfm.factored_fast_marching(
                model.velocity_model,
                dx=(model.grid_spacing, model.grid_spacing, model.grid_spacing),
                x_s=model.get_location_indices(station),
                order=2,
            )
            distances = model.get_source_distance_grid(station)
            arrival_times = eikonal_factors * distances
            station_travel_times = cls(
                center=model.center,
                velocity_model_hash=model.hash(),
                station=station,
                east_bounds=model.east_bounds,
                north_bounds=model.north_bounds,
                depth_bounds=model.depth_bounds,
                grid_spacing=model.grid_spacing,
                fmm_tracer=implementation,
            )
            station_travel_times._travel_times = arrival_times.astype(np.float32)
            if save:
                station_travel_times.save(save)

            return station_travel_times

        loop = asyncio.get_running_loop()
        work = functools.partial(eikonal_wrapper, model)

        return await loop.run_in_executor(executor, work)

    @property
    def filename(self) -> str:
        return f"{self.station.nsl.pretty}-{self.velocity_model_hash}.3dtt"

    def get_travel_time_interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            self._interpolator = RegularGridInterpolator(
                (self._east_coords, self._north_coords, self._depth_coords),
                self.travel_times,
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def _get_spline_coefficients(self) -> np.ndarray:
        # Computed once: RegularGridInterpolator's cubic method solves the spline
        # of the whole volume on every call. The odd reflection continues the
        # gradient across the edges; the spline's mirror boundary alone forces it
        # to zero there (up to 22 ms error for a travel time cone).
        if self._spline_coefficients is None:
            padded = np.pad(
                self.travel_times, SPLINE_PAD, mode="reflect", reflect_type="odd"
            )
            self._spline_coefficients = ndimage.spline_filter(
                padded, order=3, output=np.float32, mode="mirror"
            )
        return self._spline_coefficients

    def interpolate(
        self,
        offsets: np.ndarray,
        method: InterpolationMethod = "linear",
    ) -> np.ndarray:
        """Interpolate travel times at offsets from the center.

        Args:
            offsets (np.ndarray): Offsets (east, north, depth) in the grid frame
                in [m], shape (N, 3).
            method (InterpolationMethod, optional): Interpolation method.
                Defaults to "linear".

        Returns:
            np.ndarray: Travel times in [s], NaN outside the volume.
        """
        offsets = np.atleast_2d(np.asarray(offsets, dtype=float))
        coords = (self._east_coords, self._north_coords, self._depth_coords)
        start = np.array([c[0] for c in coords])
        spacing = np.array([c[1] - c[0] for c in coords])
        end = np.array([c[-1] for c in coords])
        tolerance = 1e-6 * spacing
        inside = np.all(
            (offsets >= start - tolerance) & (offsets <= end + tolerance), axis=1
        )

        travel_times = np.full(offsets.shape[0], np.nan)
        if not inside.any():
            return travel_times
        if method == "cubic":
            indices = ((offsets[inside] - start) / spacing).T + SPLINE_PAD
            travel_times[inside] = ndimage.map_coordinates(
                self._get_spline_coefficients(),
                indices,
                order=3,
                mode="mirror",
                prefilter=False,
            )
        else:
            interpolator = self.get_travel_time_interpolator()
            travel_times[inside] = interpolator(
                np.clip(offsets[inside], start, end), method=method
            )
        return travel_times

    def interpolate_travel_time(
        self,
        location: Location,
        method: InterpolationMethod = "linear",
    ) -> float:
        """Travel time to a location in [s], NaN outside the volume."""
        return float(self.interpolate(grid_offset(location, self.center), method)[0])

    async def interpolate_nodes(
        self,
        nodes: Sequence[Node],
        method: InterpolationMethod = "linear",
    ) -> np.ndarray:
        """Travel times to octree nodes in [s], NaN outside the volume."""
        offsets = node_grid_offsets(nodes, self.center)
        return await asyncio.to_thread(self.interpolate, offsets, method)

    def get_meshgrid(self) -> tuple[np.ndarray, ...]:
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
        logger.debug("saving travel times to %s", file)

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
        logger.debug("loading travel times from %s", file)
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

    def export_vtk(self, filename: Path, reference: Location | None = None) -> None:
        offset = grid_offset(reference, self.center) if reference else np.zeros(3)

        out_file = gridToVTK(
            str(filename),
            self._east_coords - offset[0],
            self._north_coords - offset[1],
            np.array((-self._depth_coords + offset[2])[::-1]),
            pointData={"travel_time": np.array(self.travel_times[:, :, ::-1])},
        )
        logger.debug(
            "vtk: exported travel times of %s to %s",
            self.station.nsl.pretty,
            out_file,
        )


class FastMarching3DTracer(RayTracer):
    """3D velocity model travel time calculation using fast marching method."""

    tracer: Literal["FastMarching3D"] = "FastMarching3D"

    phase: PhaseDescription = "fm3d:P"
    interpolation_method: InterpolationMethod = Field(
        default="linear",
        description="Interpolation method for travel times in the volume."
        " Choose from `nearest`, `linear` or `cubic`. Nodes and locations outside"
        " the volume have no travel time.",
    )
    nthreads: int = Field(
        default=0,
        description="Number of threads to use for travel time."
        " If set to `0`, `cpu_count*2` will be used.",
    )

    velocity_model: VelocityModels = Field(
        default_factory=Constant3DVelocityModel,
        description="Velocity model for the ray tracer.",
    )
    implementation: FMMImplementation = Field(
        default="eikonalfm",
        description="Implementation of the fast marching method, only `eikonalfm`"
        " (second order, factored).",
    )

    _travel_time_volumes: dict[str, StationTravelTimeVolume] = PrivateAttr({})
    _velocity_model: VelocityModel3D | None = PrivateAttr(None)

    _stations: StationList = PrivateAttr()
    _node_lut: ArrayLRUCache[bytes] = PrivateAttr()

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

    def has_travel_times_volume(self, location: Location) -> bool:
        return location.location_hash() in self._travel_time_volumes

    async def prepare(
        self,
        octree: Octree,
        stations: StationInventory,
        rundir: Path | None = None,
        use_cached: bool = True,
    ) -> None:
        logger.info("loading velocity model %s", self.velocity_model.model)

        octree.reset()
        self.velocity_model.prepare()
        velocity_model = self.velocity_model.get_model(octree, stations)

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
                offset = grid_offset(station, velocity_model.center)
                stations.exclude_station(
                    station,
                    reason=f"outside fast-marching velocity model, offset {offset}",
                )

        velocity_model.check_velocities(self.phase)
        for station in stations:
            logger.info(
                "velocity at station %s: %.1f m/s",
                station.nsl.pretty,
                velocity_model.get_velocity(station),
            )

        nodes_covered = [
            node
            for node in octree.leaf_nodes
            if velocity_model.is_inside(node.as_location())
        ]
        if not nodes_covered:
            raise ValueError("no octree node is inside the velocity model")

        logger.info(
            "%d%% of the octree nodes are inside the %s velocity model,"
            " nodes outside have no travel times",
            len(nodes_covered) / len(octree.leaf_nodes) * 100,
            self.phase,
        )

        self._stations = StationList.from_inventory(stations)
        self._node_lut = ArrayLRUCache(name="fast-marching3d", short_name="FM3D")

        cache_dir = FMM_CACHE_DIR / self.implementation / f"{velocity_model.hash()}"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        elif use_cached:
            self._load_cached_travel_times(cache_dir)

        stations_to_calculate = [
            station
            for station in self._stations
            if not self.has_travel_times_volume(station)
        ]
        await self._calculate_travel_times(stations_to_calculate, cache_dir)

        logger.debug("weeding velocity model cache")
        station_hashes = [station.location_hash() for station in stations]
        for station_hash in self._travel_time_volumes.copy():
            if station_hash not in station_hashes:
                del self._travel_time_volumes[station_hash]

        if rundir is not None:
            vtk_dir = rundir / "3d-models"
            vtk_dir.mkdir(parents=True, exist_ok=True)

            logger.info("exporting vtk files")
            velocity_model.export_vtk(
                vtk_dir / f"velocity-model-{self.phase.replace(':', '-')}",
                reference=octree.location,
            )
            for volume in self._travel_time_volumes.values():
                volume.export_vtk(
                    vtk_dir / f"travel-times-{volume.station.nsl.pretty}",
                    reference=octree.location,
                )

    def _load_cached_travel_times(self, cache_dir: Path) -> None:
        logger.debug("loading travel times volumes from cache %s...", cache_dir)
        volumes: dict[str, StationTravelTimeVolume] = {}
        for file in cache_dir.glob("*.3dtt"):
            try:
                travel_times = StationTravelTimeVolume.load(file)
            except (zipfile.BadZipFile, ValidationError, KeyError, OSError):
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
        if self._velocity_model is None:
            raise AttributeError("velocity model has not been prepared yet")
        velocity_model = self._velocity_model
        if not stations:
            return

        nthreads = self.nthreads if self.nthreads > 0 else get_cpu_count() * 2
        executor = ThreadPoolExecutor(
            max_workers=nthreads,
            thread_name_prefix="qseek-fmm3d",
        )

        async def worker_station_travel_time(station: Station) -> None:
            volume = await StationTravelTimeVolume.calculate_from_eikonal(
                velocity_model,
                station,
                save=cache_dir,
                executor=executor,
                implementation=self.implementation,
            )
            self.add_travel_time_volume(station, volume)

        start = datetime_now()
        tasks = [
            asyncio.create_task(worker_station_travel_time(station))
            for station in stations
        ]
        try:
            with get_progress() as progress:
                status = progress.add_task(
                    f"calculating travel time volumes for {len(tasks)} stations"
                    f" ({nthreads} threads)",
                    total=len(tasks),
                )
                progress.update(status, visible=False)
                for task in asyncio.as_completed(tasks):
                    await task
                    progress.advance(status)
                progress.remove_task(status)
        finally:
            # On a failure, cancel the pending volumes and free the threads
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            executor.shutdown(wait=False, cancel_futures=True)

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

    @alog_call
    async def get_travel_times(
        self,
        phase: str,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        if phase != self.phase:
            raise ValueError(f"phase {phase} is not supported by this tracer")

        station_indices = self._stations.get_indices(stations)

        node_lut = self._node_lut
        try:
            travel_times = [node_lut[node.hash][station_indices] for node in nodes]
            return np.array(travel_times)
        except KeyError:
            fill_nodes = [node for node in nodes if node.hash not in node_lut]
            await self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_travel_times(phase, nodes, stations)

    async def fill_lut(self, nodes: Sequence[Node]) -> None:
        n_nodes = len(nodes)
        volumes = [self.get_travel_time_volume(station) for station in self._stations]
        # All volumes of a tracer share the center of its velocity model
        offsets: dict[str, np.ndarray] = {}
        for volume in volumes:
            key = volume.center.location_hash()
            if key not in offsets:
                offsets[key] = node_grid_offsets(nodes, volume.center)

        def interpolate(advance: Callable[[], Any] | None = None) -> np.ndarray:
            travel_times = np.empty((n_nodes, len(volumes)))
            for idx, volume in enumerate(volumes):
                travel_times[:, idx] = volume.interpolate(
                    offsets[volume.center.location_hash()],
                    method=self.interpolation_method,
                )
                if advance is not None:
                    advance()
            return travel_times

        # One thread for all stations; a thread per station costs more than the
        # interpolation of a refinement batch
        if n_nodes < PROGRESS_MIN_NODES:
            travel_times = await asyncio.to_thread(interpolate)
        else:
            with get_progress() as progress:
                status = progress.add_task(
                    f"interpolating {self.phase} travel times for {n_nodes} nodes",
                    total=len(volumes),
                )
                travel_times = await asyncio.to_thread(
                    interpolate, lambda: progress.advance(status)
                )
                progress.remove_task(status)

        node_lut = self._node_lut
        for node, station_travel_times in zip(nodes, travel_times, strict=True):
            node_lut[node.hash] = station_travel_times

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
        for travel_time, _receiver in zip(traveltimes, receivers, strict=True):
            # Outside the velocity model
            if np.isnan(travel_time):
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=travel_time)
            arrival = ModelledArrival(
                time=arrivaltime,
                phase=phase,
            )
            arrivals.append(arrival)
        return arrivals
