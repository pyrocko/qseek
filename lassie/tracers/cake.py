from __future__ import annotations

import logging
import struct
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np
from lmdb import Environment
from pydantic import BaseModel, PrivateAttr, root_validator
from pyrocko import spit
from pyrocko.cake import LayeredModel, PhaseDef, Ray, m2d, read_nd_model_str

from lassie.models.station import Station, Stations
from lassie.octree import Node, Octree
from lassie.tracers.base import RayTracer
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from lmdb import _Database

logger = logging.getLogger(__name__)

MAX_DBS = 16
EXAMPLE_MODEL = """
   0.00      5.50    3.59    2.7
   1.00      5.50    3.59    2.7
   1.00      6.00    3.92    2.7
   4.00      6.00    3.92    2.7
   4.00      6.20    4.05    2.7
   8.00      6.20    4.05    2.7
   8.00      6.30    4.12    2.7
   13.00     6.30    4.12    2.7
   13.00     6.40    4.18    2.7
   17.00     6.40    4.18    2.7
   17.00     6.50    4.25    2.7
   22.00     6.50    4.25    2.7
   22.00     6.60    4.31    2.7
   26.00     6.60    4.31    2.7
   26.00     6.80    4.44    2.7
   30.00     6.80    4.44    2.7
   30.00     8.10    5.29    2.7
   45.00     8.10    5.29    2.7
   45.00     8.50    5.56    2.7
   71.00     8.50    5.56    2.7
   71.00     8.73    5.71    2.7
   101.00    8.73    5.71    2.7
   101.00    8.73    5.71    2.7
   201.00    8.73    5.71    2.7
   201.00    8.73    5.71    2.7
   301.00    8.73    5.71    2.7
   301.00    8.73    5.71    2.7
   401.00    8.73    5.71    2.7
"""


class SPTreeModel(BaseModel):
    earthmodel: str

    source_bounds: tuple[float, float]
    distance_bounds: tuple[float, float]
    receiver_depth_bounds: tuple[float, float]
    time_tolerance: float
    spatial_tolerance: float

    filename: Path

    def get_sptree(self) -> spit.SPTree:
        ...

    def save_sptree(self) -> None:
        ...

    def calculate_tree(self) -> spit.SPTree:
        ...


class SPTreeModels(BaseModel):
    __root__: list[SPTreeModel] = []

    def save_model(self, model: SPTreeModel) -> None:
        self.__root__.append(model)

    def get_model(self, model: SPTreeModel) -> spit.SPTree:
        for model in self:
            ...

    def __iter__(self) -> Iterator[SPTreeModel]:
        return iter(self.__root__)


def hash_to_bytes(hash: int) -> bytes:
    return struct.pack("q", hash)


class Timing(BaseModel):
    definition: str

    def as_phase_def(self) -> PhaseDef:
        return PhaseDef(definition=self.definition)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[PhaseDescription, Timing] = {
        "cake:P": Timing(definition="P"),
        "cake:S": Timing(definition="S"),
    }
    earthmodel: str = EXAMPLE_MODEL

    cache: Path = Path("/tmp/test-lmdb")
    cache_hits: int = 0

    _earthmodel: LayeredModel = PrivateAttr()
    _earthmodel_hash: bytes = PrivateAttr()
    _lmdb: Environment = PrivateAttr()

    @root_validator
    def _read_earthmodel(cls, values):  # noqa N805
        try:
            LayeredModel.from_scanlines(read_nd_model_str(values.get("earthmodel")))
        except Exception as exc:
            raise ValueError("bad earthmodel") from exc
        return values

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self._earthmodel = LayeredModel.from_scanlines(
            read_nd_model_str(self.earthmodel)
        )
        data = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            self._earthmodel.profile(param).dump(data)
        self._earthmodel_hash = sha1(data.getvalue()).digest()

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.timings.keys())

    def get_vmin(self) -> float:
        earthmodel = self._earthmodel
        vs = np.concatenate((earthmodel.profile("vp"), earthmodel.profile("vs")))
        return float((vs[vs != 0.0]).min())

    def get_filename_hash(self, phase: PhaseDescription) -> str:
        return f"{phase}-{self._earthmodel_hash}.dat"

    def prepare(self, octree: Octree, stations: Stations) -> None:
        for phase in self.get_available_phases():
            try:
                self.get_traveltime_tree(phase)
            except KeyError:
                self.precalculate_traveltimes(phase, octree, stations)

    def precalculate_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> None:
        if phase not in self.timings:
            raise ValueError(f"timing {phase} is not defined")
        phase_def = self.timings[phase].as_phase_def()

        logger.info("pre-calculating traveltimes for phase %s", phase)

        distances = octree.distances_stations(stations)
        source_depths = np.array(octree.depth_bounds)
        receiver_depths = np.fromiter((sta.effective_depth for sta in stations), float)

        x_bounds = [
            [receiver_depths.min(), receiver_depths.max()],
            [source_depths.min(), source_depths.max()],
            [distances.min(), distances.max()],
        ]
        x_tolerance = [octree.size_limit, octree.size_limit, octree.size_limit]
        # TODO: Time tolerance is too hardcoded
        t_tolerance = octree.size_initial / (self.get_vmin() * 5.0)

        def evaluate(args) -> float | None:
            receiver_depth, source_depth, distances = args
            rays = self._earthmodel.arrivals(
                phases=phase_def,
                distances=[distances * m2d],
                zstart=source_depth,
                zstop=receiver_depth,
            )
            times = np.fromiter((ray.t for ray in rays), float)
            return times.min() if times.size else None

        traveltime_tree = spit.SPTree(
            f=evaluate,
            ftol=np.array(t_tolerance),
            xbounds=np.array(x_bounds),
            xtols=x_tolerance,
        )
        filename = self.get_filename_hash(phase)
        logger.debug("writing SPTree to %s", filename)
        traveltime_tree.dump(filename)

    def get_traveltime_tree(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> spit.SPTree:
        ...

    def get_traveltime(self, phase: str, node: Node, station: Station) -> float:
        if phase not in self.timings:
            raise ValueError(f"timing {phase} is not defined.")
        phase_def = self.timings[phase].as_phase_def()

        distance = node.distance_station(station)
        rays: list[Ray] = self._earthmodel.arrivals(
            distances=[distance * m2d],
            phases=phase_def,
            zstart=node.depth,
        )
        if not rays:
            return np.nan
        return rays[0].t

    def get_traveltimes(
        self, phase: str, octree: Octree, stations: Stations
    ) -> np.ndarray:
        if phase not in self.timings:
            raise ValueError(f"timing {phase} is not defined.")
        timing = self.timings[phase].as_phase_def()

        distances = octree.distances_receivers(stations)
        return np.array(
            [
                self._get_node_traveltimes(timing, node, distance)
                for node, distance in zip(octree, distances)
            ]
        )

    def _get_lmdb_database(self, timing: PhaseDef) -> _Database:
        hash = sha1(self._earthmodel_hash)
        if not timing._definition:
            raise AttributeError("PhaseDef has no definition")
        hash.update(timing._definition.encode())
        return self._lmdb.open_db(hash.digest())

    def _get_node_traveltimes(
        self, timing: PhaseDef, node: Node, distances: np.ndarray
    ) -> np.ndarray:
        database = self._get_lmdb_database(timing)

        hash = sha1(distances.tobytes())
        hash.update(hash_to_bytes(node.__hash__()))
        node_key = hash.digest()

        with self._lmdb.begin() as txn:
            traveltimes = txn.get(node_key, db=database)
            if traveltimes:
                self.cache_hits += 1
                print(f"hit cache {self.cache_hits}")
                return np.frombuffer(traveltimes, dtype=np.float32)

        traveltimes = self._calculate_node_traveltimes(timing, node, distances)
        with self._lmdb.begin(write=True) as txn:
            txn.put(node_key, traveltimes.tobytes(), db=database)
        return traveltimes

    def _calculate_node_traveltimes(
        self, timing: PhaseDef, node: Node, distances: np.ndarray
    ) -> np.ndarray:
        print("calculating traveltimes")
        traveltimes = np.full_like(distances, np.nan, dtype=np.float32)
        for idx, distance in enumerate(distances):
            rays: list[Ray] = self._earthmodel.arrivals(
                distances=[distance * m2d],
                phases=timing,
                zstart=node.depth,
            )
            if rays:
                traveltimes[idx] = rays[0].t
        return traveltimes
