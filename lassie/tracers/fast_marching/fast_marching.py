import asyncio
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence

import numpy as np
from pydantic import BaseModel, PrivateAttr
from pyrocko.modelling import eikonal
from scipy.interpolate import RegularGridInterpolator

from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    VelocityModel3D,
    VelocityModels,
)
from lassie.utils import CACHE_DIR, PhaseDescription, log_call

if TYPE_CHECKING:
    from lassie.models.location import Location
    from lassie.models.station import Station, Stations
    from lassie.octree import Octree


FMM_CACHE_DIR = CACHE_DIR / "fast-marching-cache"
logger = logging.getLogger(__name__)


class FastMarchingArrival(ModelledArrival):
    tracer: Literal["FastMarchingArrival"] = "FastMarchingArrival"
    phase: PhaseDescription


class StationTravelTimes3D(BaseModel):
    origin: Location
    station: Station

    velocity_model_hash: str

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]
    grid_spacing: float

    travel_times: np.ndarray | None = None

    north_coords: np.ndarray = PrivateAttr(None)
    east_coords: np.ndarray = PrivateAttr(None)
    depth_coords: np.ndarray = PrivateAttr(None)

    # Cached values
    _file: Path | None = PrivateAttr(None)
    _interpolator: RegularGridInterpolator | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self.east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self.north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1] + grid_spacing,
            grid_spacing,
        )
        self.depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1] + grid_spacing,
            grid_spacing,
        )

    @classmethod
    async def calculate_from_eikonal(
        cls,
        model: VelocityModel3D,
        station: Station,
    ) -> Self:
        arrival_times = model.get_source_arrival_grid(station)

        await asyncio.to_thread(
            eikonal.eikonal_solver_fmm_cartesian,
            model.velocity_model,
            arrival_times,
            delta=model.grid_spacing,
        )

        return cls(
            origin=model.origin,
            velocity_model_hash=model.hash(),
            station=station,
            travel_times=arrival_times,
            east_bounds=model.east_bounds,
            north_bounds=model.north_bounds,
            depth_bounds=model.depth_bounds,
            grid_spacing=model.grid_spacing,
        )

    @property
    def filename(self) -> str:
        return f"{self.station.pretty_nsl}-{self.velocity_model_hash}.3dtt"

    def trim_to_octree(self, octree: Octree) -> None:
        east_idx = np.where(
            (self.east_coords >= octree.east_bounds[0])
            & (self.east_coords <= octree.east_bounds[1])
        )[0]
        north_idx = np.where(
            (self.north_coords >= octree.north_bounds[0])
            & (self.north_coords <= octree.north_bounds[1])
        )[0]
        depth_idx = np.where(
            (self.depth_coords >= octree.effective_depth_bounds[0])
            & (self.depth_coords <= octree.effective_depth_bounds[1])
        )[0]

        self.arrival_times = self.arrival_times[east_idx, north_idx, depth_idx]
        self.east_bounds = (
            self.east_coords[east_idx][0],
            self.east_coords[east_idx][1],
        )
        self.north_bounds = (
            self.north_coords[north_idx][0],
            self.north_coords[north_idx][1],
        )
        self.depth_bounds = (
            self.depth_coords[depth_idx][0],
            self.depth_coords[depth_idx][1],
        )

    def get_traveltime_interpolator(
        self, method: Literal["linear", "nearest"] = "linear"
    ) -> RegularGridInterpolator:
        if self.travel_times is None:
            self.travel_times = self._load_travel_times()

        if self._interpolator is None:
            self._interpolator = RegularGridInterpolator(
                (self.east_coords, self.north_coords, self.depth_coords),
                self.travel_times,
                method=method,
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def interpolate_travel_time(self, location: Location) -> float:
        interpolator = self.get_traveltime_interpolator()
        offset = location.offset_to(self.origin)

        return interpolator([offset])

    def interpolate_travel_times(self, octree: Octree) -> np.ndarray:
        interpolator = self.get_traveltime_interpolator()

        coordinates = []
        for node in octree:
            location = node.as_location()
            coordinates.append(location.offset_to(self.origin))

        return interpolator(coordinates)

    def save(self, path: Path) -> Path:
        """Save travel times to a zip file.

        The zip file contains a model.json file with the model metadata and a
        model.sptree file with the travel times.

        Args:
            path (Path): path to save the travel times to

        Returns:
            Path: path to the saved travel times
        """
        if self.travel_times is None:
            raise AttributeError("travel times have not been calculated yet")
        file = path / self.filename if path.is_dir() else path
        logger.info("saving travel times to %s...", file)

        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr("model.json", self.model_dump_json(indent=2))
            with NamedTemporaryFile() as tmpfile:
                self.travel_times.tofile(tmpfile.name)
                archive.write(tmpfile.name, "traveltimes.npz")
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        """Load 3D travel times from a zip file.

        Args:
            file (Path): path to the zip file containing the travel times

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

        with zipfile.ZipFile(
            self._file, "r"
        ) as archive, TemporaryDirectory() as temp_dir:
            archive.extract("model.sptree", path=temp_dir)
            return np.load(str(Path(temp_dir) / "traveltimes.npz"))


class FastMarchingPhaseTracer(BaseModel):
    velocity_model: VelocityModels = Constant3DVelocityModel(
        velocity=3000.0, grid_spacing="auto"
    )
    interpolation_method: Literal["linear", "nearest"] = "nearest"

    _traveltime_models: dict[int, StationTravelTimes3D] = PrivateAttr({})

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        velocity_model = self.velocity_model.get_model(octree, stations)

        cache_dir = FMM_CACHE_DIR / f"{velocity_model.hash()}"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        for file in cache_dir.glob("*.3dtt"):
            travel_times = StationTravelTimes3D.load(file)
            self._traveltime_models[hash(travel_times.station)] = travel_times

        logger.info(
            "using %d travel times volumes from cache", len(self._traveltime_models)
        )

        for station in stations:
            if station.pretty_nsl in self._traveltime_models:
                continue

            logger.info("pre-calculating traveltimes for %s...", station.pretty_nsl)
            travel_times = await StationTravelTimes3D.calculate_from_eikonal(
                velocity_model, station
            )
            travel_times.trim_to_octree(octree)
            travel_times.save(cache_dir)

            self._traveltime_models[hash(station)] = travel_times

    def get_travel_time(self, source: Location, receiver: Location) -> float:
        station_travel_times = self._traveltime_models[hash(receiver)]
        return station_travel_times.interpolate_travel_time(source)

    def get_travel_times(self, octree: Octree, stations: Stations) -> np.ndarray:
        result = []
        for station in stations:
            station_travel_times = self._traveltime_models[hash(station)]
            result.append(station_travel_times.interpolate_travel_times(octree))
        return np.array(result).T


class FastMarchingRayTracer(RayTracer):
    tracer: Literal["FastMarchingRayTracer"] = "FastMarchingRayTracer"

    tracers: dict[PhaseDescription, FastMarchingPhaseTracer] = {
        "fm:P": FastMarchingPhaseTracer(),
        "fm:S": FastMarchingPhaseTracer(),
    }

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        for tracer in self.tracers.values():
            await tracer.prepare(octree, stations)

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.tracers.keys())

    def clear_cache(self) -> None:
        logger.info("clearing fast marching cache...")
        for file in FMM_CACHE_DIR.glob("*/*.3dtt"):
            file.unlink()
        for dir in FMM_CACHE_DIR.glob("*"):
            dir.rmdir()

    def _get_tracer(self, phase: str) -> FastMarchingPhaseTracer:
        return self.tracers[phase]

    def get_travel_time_location(
        self, phase: str, source: Location, receiver: Location
    ) -> float:
        if phase not in self.tracers:
            raise ValueError(f"Phase {phase} is not defined.")
        return self._get_tracer(phase).get_travel_time(source, receiver)

    @log_call
    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        return self._get_tracer(phase).get_travel_times(octree, stations)

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
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
