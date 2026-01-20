from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import skfmm
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pyrocko.modelling import eikonal
from scipy.interpolate import RegularGridInterpolator

from qseek.cache_lru import ArrayLRUCache
from qseek.models.layered_model import LayeredModel
from qseek.models.station import Station, StationInventory, StationList
from qseek.octree import Node
from qseek.stats import get_progress
from qseek.tracers.base import ModelledArrival, RayTracer
from qseek.tracers.utils import LayeredEarthModel1D, surface_distances_reference
from qseek.utils import Range, alog_call, datetime_now, get_cpu_count

if TYPE_CHECKING:
    from qseek.models.location import Location
    from qseek.octree import Octree

logger = logging.getLogger(__name__)
Phase = Literal["fm:P", "fm:S"]

InterpolationMethod = Literal["nearest", "linear", "cubic"]
FMMImplementation = Literal["pyrocko", "scikit-fmm"]


class StationTravelTimeTable(BaseModel):
    station: Station
    phase: Phase

    distance_max: float
    depth_range: Range = Field(
        ...,
        description="Depth range for the travel time table. Relative to the station.",
    )
    grid_spacing: float = Field(..., ge=0.0)

    earth_model: LayeredModel

    created: datetime = Field(default_factory=datetime_now)

    _distances: np.ndarray = PrivateAttr()
    _depths: np.ndarray = PrivateAttr()

    _travel_times: np.ndarray | None = PrivateAttr(None)
    _interpolator: RegularGridInterpolator | None = PrivateAttr(None)

    @field_validator("depth_range", mode="after")
    @classmethod
    def check_depth_range(cls, depth_range: Range) -> Range:
        if depth_range.start > 0.0:
            raise ValueError("depth range start must be >= 0.0")
        return depth_range

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self._distances = np.arange(
            0.0,
            self.distance_max + grid_spacing,
            grid_spacing,
        )
        offset = self.depth_range.start % grid_spacing
        self._depths = np.arange(
            self.depth_range.start - offset,
            self.depth_range.end + grid_spacing,
            grid_spacing,
        )

    def _get_velocity_grid(self) -> np.ndarray:
        if self.phase == "fm:P":
            velocities = self.earth_model.vp_interpolator
        elif self.phase == "fm:S":
            velocities = self.earth_model.vs_interpolator
        else:
            raise ValueError(f"unknown phase {self.phase}")
        vels = velocities(self._depths + self.station.effective_depth)
        return np.tile(vels, self._distances.size).reshape(
            (self._distances.size, self._depths.size)
        )

    def _get_source_arrival_grid(self) -> np.ndarray:
        times = np.full(
            shape=(self._distances.size, self._depths.size),
            fill_value=-1.0,
        )
        depth_idx = np.where(self._depths == 0.0)
        if depth_idx[0].size != 1:
            raise ValueError("depth 0.0 not found in depth range")
        times[0, depth_idx[0]] = 0.0
        return times

    def _get_travel_time_interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            if self._travel_times is None:
                raise ValueError("travel times not calculated yet")
            self._interpolator = RegularGridInterpolator(
                (self._distances, self._depths),
                self._travel_times,
                bounds_error=True,
                fill_value=None,
            )
        return self._interpolator

    async def calculate(
        self,
        implementation: FMMImplementation = "scikit-fmm",
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        velocity_model = self._get_velocity_grid()
        arrival_times = self._get_source_arrival_grid()

        def eikonal_wrapper(arrival_times: np.ndarray, delta: float) -> np.ndarray:
            logger.debug(
                "calculating travel time table %s for %s, grid size %s,"
                " spacing %s m using %s...",
                self.phase,
                self.station.nsl.pretty,
                arrival_times.shape,
                self.grid_spacing,
                implementation,
            )
            if implementation == "pyrocko":
                eikonal.eikonal_solver_fmm_cartesian(
                    velocity_model,
                    arrival_times,
                    delta=delta,
                )
            elif implementation == "scikit-fmm":
                arrival_times = skfmm.travel_time(
                    arrival_times,
                    velocity_model,
                    dx=delta,
                    order=2,
                )
            else:
                raise ValueError(f"unknown eikonal solver {implementation}")
            self._travel_times = arrival_times
            return arrival_times

        loop = asyncio.get_running_loop()
        work = functools.partial(
            eikonal_wrapper,
            arrival_times,
            delta=self.grid_spacing,
        )
        start_time = time.time()
        await loop.run_in_executor(executor, work)
        logger.debug(
            "calculated travel time table %s for %s in %f seconds",
            self.phase,
            self.station.nsl.pretty,
            time.time() - start_time,
        )

    def get_travel_time(
        self,
        source_distance: float,
        source_depth: float,
        method: InterpolationMethod = "linear",
    ) -> float:
        interpolator = self._get_travel_time_interpolator()
        traveltime = interpolator([source_distance, source_depth], method=method)
        return float(traveltime.squeeze())

    async def get_travel_times(
        self,
        surface_offsets: np.ndarray,
        depth_offsets: np.ndarray,
        method: InterpolationMethod = "linear",
    ) -> np.ndarray:
        interpolator = self._get_travel_time_interpolator()
        try:
            return await asyncio.to_thread(
                interpolator,
                np.asarray([surface_offsets, depth_offsets]).T,
                method=method,
            )
        except ValueError as e:
            raise ValueError(
                "error interpolating travel times: "
                f"distances {surface_offsets.min()}-{surface_offsets.max()} m, "
                f"depths {depth_offsets.min()}-{depth_offsets.max()} m, "
                f"table distance max {self.distance_max} m, "
                f"depth range {self.depth_range} "
                f"for station {self.station.nsl.pretty} and phase {self.phase}"
            ) from e


class FastMarchingTracer(RayTracer):
    """Travel time calculation for layered 1D velocity models.

    Calculation is based on fast marching method to solve the Eikonal solution.
    This the more perfomant than the Pyrocko Cake ray tracer, especially for
    large number of stations and nodes.
    """

    tracer: Literal["FastMarching"] = "FastMarching"

    velocity_model: LayeredEarthModel1D | None = Field(
        default_factory=LayeredEarthModel1D,
        description="Velocity model for the ray tracer.",
    )

    interpolation_method: InterpolationMethod = Field(
        default="linear",
        description="Interpolation method for travel times in the volume."
        " Choose from `nearest`, `linear` or `cubic`.",
    )
    nthreads: int = Field(
        default=0,
        description="Number of threads to use for travel time."
        " If set to `0`, `cpu_count*2` will be used.",
    )

    implementation: FMMImplementation = Field(
        default="pyrocko",
        description="Implementation of the Fast Marching Method. Pyrocko only supports"
        " first-order FMM for now.",
    )

    phases: tuple[Phase, ...] = Field(
        default=("fm:P", "fm:S"),
        description="Phases to calculate.",
    )
    _travel_time_tables: dict[tuple[str, Phase], StationTravelTimeTable] = PrivateAttr(
        default_factory=dict
    )

    _octree: Octree = PrivateAttr()
    _stations: StationList = PrivateAttr()
    _layered_model: LayeredModel | None = PrivateAttr(None)
    _node_lut: ArrayLRUCache[tuple[bytes, Phase]] = PrivateAttr()

    def get_travel_time_table(
        self,
        location: Location,
        phase: Phase,
    ) -> StationTravelTimeTable:
        """Get the travel time table for a given station and phase."""
        key = (location.location_hash(), phase)
        if key not in self._travel_time_tables:
            raise ValueError(
                f"travel time table for {location} and phase {phase} not found"
            )
        return self._travel_time_tables[key]

    def get_available_phases(self) -> tuple[str, ...]:
        return self.phases

    def set_layered_model(self, model: LayeredModel) -> None:
        self._layered_model = model

    def get_layered_model(self) -> LayeredModel:
        if self._layered_model is None:
            raise ValueError("layered model is not set")
        return self._layered_model

    def model_post_init(self, context: Any) -> None:
        if self.velocity_model is not None:
            self._layered_model = LayeredModel.from_earth_model(self.velocity_model)

    async def prepare(
        self,
        octree: Octree,
        stations: StationInventory,
        rundir: Path | None = None,
    ) -> None:
        """Prepare the tracer for a given set of stations."""
        logger.debug("preparing FastMarchingTracer...")
        if self._layered_model is None:
            raise ValueError("layered model must be set for FastMarchingTracer")

        self._stations = StationList.from_inventory(stations)
        self._octree = octree

        if rundir:
            export_dir = rundir / "velocity-model"
            export_dir.mkdir(exist_ok=True, parents=True)
            self._layered_model.plot(
                depth_range=octree.effective_depth_bounds,
                export=export_dir / "layered_model.png",
            )

        await self._calculate_travel_times()

        self._node_lut = ArrayLRUCache(name="fast-marching", short_name="FM")
        nodes = octree.leaf_nodes
        for phase in self.phases:
            logger.info(
                "warming up traveltime LUT %s for %d stations and %d nodes",
                phase,
                stations.n_stations,
                len(nodes),
            )
            await self.fill_lut(nodes, phase)

    def add_travel_time_table(self, table: StationTravelTimeTable) -> None:
        """Add a travel time slice to the tracer."""
        key = (table.station.location_hash(), table.phase)
        if key in self._travel_time_tables:
            logger.warning(
                "duplicate travel time table for location %.4f %.4f",
                table.station.effective_lat,
                table.station.effective_lon,
            )
        self._travel_time_tables[key] = table

    async def _calculate_travel_times(self) -> None:
        if self._layered_model is None:
            raise ValueError("layered model must be set for FastMarchingTracer")

        nthreads = self.nthreads if self.nthreads > 0 else get_cpu_count() * 2
        executor = ThreadPoolExecutor(
            max_workers=nthreads,
            thread_name_prefix="qseek-fmm",
        )

        octree = self._octree
        octree_corners = octree.get_corners()

        async def worker_station_travel_time(station: Station, phase: Phase) -> None:
            surface_distances = np.array(
                [station.surface_distance_to(corner) for corner in octree_corners]
            )
            octree_depth_range = (
                np.array(octree.depth_bounds) - octree.location.effective_elevation
            )
            depth_margin = octree.depth_bounds.width() * 0.01  # 1% margin

            volume = StationTravelTimeTable(
                station=station,
                phase=phase,
                distance_max=surface_distances.max() * 1.01,  # 1% margin
                depth_range=Range(
                    min(octree_depth_range[0] - station.effective_depth, 0.0)
                    - depth_margin,
                    octree_depth_range[1] + station.effective_elevation + depth_margin,
                ),
                grid_spacing=octree.smallest_node_size(),
                earth_model=self._layered_model,  # type: ignore
            )

            await volume.calculate(
                implementation=self.implementation,
                executor=executor,
            )
            self.add_travel_time_table(volume)

        eikonal_work = [
            worker_station_travel_time(station, phase)
            for station in self._stations
            for phase in self.phases
        ]
        tasks = [asyncio.create_task(work) for work in eikonal_work]
        start_time = time.time()
        await asyncio.gather(*tasks)
        logger.info("calculated travel time tables in %.4f", time.time() - start_time)

    async def fill_lut(self, nodes: Sequence[Node], phase: Phase) -> None:
        travel_times = []
        n_nodes = len(nodes)

        surface_offsets = surface_distances_reference(
            nodes, self._stations, self._octree.location
        ).T
        node_depths = (
            np.array([n.depth for n in nodes]) + self._octree.location.effective_depth
        )

        with get_progress() as progress:
            status = progress.add_task(
                f"interpolating travel times for {n_nodes} nodes",
                total=self._stations.n_stations,
            )
            for idx, station in enumerate(self._stations):
                table = self.get_travel_time_table(station, phase)
                travel_times.append(
                    await table.get_travel_times(
                        surface_offsets=surface_offsets[idx],
                        depth_offsets=node_depths - station.effective_depth,
                        method=self.interpolation_method,
                    )
                )
                progress.advance(status)
            travel_times = np.array(travel_times).T
            progress.remove_task(status)

        node_lut = self._node_lut
        for node, station_travel_times in zip(nodes, travel_times, strict=True):
            node_lut[node.hash, phase] = station_travel_times

    def get_travel_time_location(
        self,
        phase: Phase,
        source: Location,
        receiver: Location,
    ):
        travel_time_table = self.get_travel_time_table(receiver, phase)
        source_distance = receiver.surface_distance_to(source)
        source_depth = source.effective_depth - receiver.effective_depth
        return travel_time_table.get_travel_time(
            source_distance,
            source_depth,
            method=self.interpolation_method,
        )

    @alog_call
    async def get_travel_times(
        self,
        phase: Phase,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        """Get travel times for a phase from a source to a set of stations.

        Args:
            phase: Phase name.
            nodes: Nodes to get traveltime for.
            stations: Stations to calculate travel times to.

        Returns:
            np.ndarray: Array of shape (n_nodes, n_stations) with travel times in
                seconds.
        """
        if phase not in self.phases:
            raise ValueError(f"phase {phase} is not supported by this tracer")

        node_lut = self._node_lut
        station_indices = self._stations.get_indices(stations)

        try:
            travel_times = [node_lut[nd.hash, phase][station_indices] for nd in nodes]
            return np.array(travel_times)
        except KeyError:
            fill_nodes = [nd for nd in nodes if (nd.hash, phase) not in node_lut]
            await self.fill_lut(fill_nodes, phase)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_travel_times(phase, nodes, stations)

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
            if traveltime is None:
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = ModelledArrival(
                time=arrivaltime,
                phase=phase,
            )
            arrivals.append(arrival)
        return arrivals
