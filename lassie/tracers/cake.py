from __future__ import annotations

import logging
import struct
import zipfile
from functools import cached_property
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from lmdb import Environment
from pydantic import BaseModel, PositiveFloat, PrivateAttr, root_validator
from pyrocko import spit
from pyrocko.cake import LayeredModel, PhaseDef, Ray, m2d, read_nd_model_str
from pyrocko.gf import meta

from lassie.tracers.base import RayTracer
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from lmdb import _Database

    from lassie.models.location import Location
    from lassie.models.station import Station, Stations
    from lassie.octree import Node, Octree

logger = logging.getLogger(__name__)

MAX_DBS = 16


class EarthModel(BaseModel):
    __root__: list[tuple[float, PositiveFloat, PositiveFloat, PositiveFloat]] = [
        (0.00, 5.50, 3.59, 2.7),
        (1.00, 5.50, 3.59, 2.7),
        (1.00, 6.00, 3.92, 2.7),
        (4.00, 6.00, 3.92, 2.7),
        (4.00, 6.20, 4.05, 2.7),
        (8.00, 6.20, 4.05, 2.7),
        (8.00, 6.30, 4.12, 2.7),
        (13.00, 6.30, 4.12, 2.7),
        (13.00, 6.40, 4.18, 2.7),
        (17.00, 6.40, 4.18, 2.7),
        (17.00, 6.50, 4.25, 2.7),
        (22.00, 6.50, 4.25, 2.7),
        (22.00, 6.60, 4.31, 2.7),
        (26.00, 6.60, 4.31, 2.7),
        (26.00, 6.80, 4.44, 2.7),
        (30.00, 6.80, 4.44, 2.7),
        (30.00, 8.10, 5.29, 2.7),
        (45.00, 8.10, 5.29, 2.7),
    ]

    class Config:
        keep_untouched = (cached_property,)

    def as_layered_model(self) -> LayeredModel:
        line_tpl = "{} {} {} {}"
        earthmodel = "\n".join(line_tpl.format(*layer) for layer in self.__root__)
        return LayeredModel.from_scanlines(read_nd_model_str(earthmodel))

    @cached_property
    def hash(self) -> str:
        layered_model = self.as_layered_model()
        model_serialised = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            layered_model.profile(param).dump(model_serialised)
        return sha1(model_serialised.getvalue()).hexdigest()


class Timing(BaseModel):
    definition: str = "P,p"

    def as_phase_defs(self) -> list[PhaseDef]:
        return [PhaseDef(definition=phase) for phase in self.definition.split(",")]

    def as_pyrocko_timing(self) -> meta.Timing:
        return meta.Timing(f"{{stored:{self.id}}}")

    @property
    def id(self) -> str:
        # TODO: use regex
        return self.definition.replace(",", "")


class SPTreeModel(BaseModel):
    earthmodel: EarthModel
    timing: Timing

    distance_bounds: tuple[float, float]
    source_depth_bounds: tuple[float, float]
    receiver_depth_bounds: tuple[float, float]
    time_tolerance: float
    spatial_tolerance: float

    _sptree: spit.SPTree | None = PrivateAttr(None)
    _file: Path | None = PrivateAttr(None)

    def calculate_tree(self) -> spit.SPTree:
        layered_model = self.earthmodel.as_layered_model()

        def evaluate(args) -> float | None:
            receiver_depth, source_depth, distances = args
            rays = layered_model.arrivals(
                phases=self.timing.as_phase_defs(),
                distances=[distances * m2d],
                zstart=source_depth,
                zstop=receiver_depth,
            )
            times = np.fromiter((ray.t for ray in rays), float)
            print(args, times)
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
        earthmodel: EarthModel,
        timing: Timing,
        distance_bounds: tuple[float, float],
        source_depth_bounds: tuple[float, float],
        receiver_depth_bounds: tuple[float, float],
        time_tolerance: float,
        spatial_tolerance: float,
    ) -> bool:
        def check_bounds(self, requested) -> bool:
            return self[0] <= requested[0] and self[1] >= requested[1]

        return (
            self.earthmodel == earthmodel
            and self.timing == timing
            and check_bounds(self.distance_bounds, distance_bounds)
            and check_bounds(self.source_depth_bounds, source_depth_bounds)
            and check_bounds(self.receiver_depth_bounds, receiver_depth_bounds)
            and self.time_tolerance <= time_tolerance
            and self.spatial_tolerance <= spatial_tolerance
        )

    @property
    def filename(self) -> Path:
        return Path(f"{self.timing.id}-{self.earthmodel.hash}")

    @classmethod
    def new(cls, **data) -> Self:
        model = cls(**data)
        model._sptree = model.calculate_tree()
        return model

    def save(self, folder: Path) -> Path:
        file = folder / self.filename
        logger.debug("saving model to %s", file)
        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr("model.json", self.json(indent=2))
            with NamedTemporaryFile() as tmpfile:
                self.get_sptree().dump(tmpfile.name)
                archive.write(tmpfile.name, "model.sptree")
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        logger.debug("loading model from %s", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model = cls.parse_raw((path / "model.json").read_bytes())
        model._file = file
        return model

    def _load_tree(self) -> spit.SPTree:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with zipfile.ZipFile(self._file, "r") as archive:
            with TemporaryDirectory() as temp_dir:
                archive.extract("model.sptree", path=temp_dir)
                return spit.SPTree(filename=str(Path(temp_dir) / "model.sptree"))

    def get_sptree(self) -> spit.SPTree:
        if self._sptree is None:
            self._sptree = self._load_tree()
        return self._sptree

    def get_traveltime(self, source: Location, receiver: Location) -> float:
        coords = [
            receiver.effective_depth,
            source.effective_depth,
            receiver.distance_to(source),
        ]
        print(coords)
        tree = self.get_sptree()
        timing = self.timing.as_pyrocko_timing()
        traveltime = timing.evaluate(
            lambda phase: tree.interpolate_many, np.array([coords])
        )
        return float(traveltime)

    def get_traveltimes(self, octree: Octree, stations: Stations):
        ...


def hash_to_bytes(hash: int) -> bytes:
    return struct.pack("q", hash)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[PhaseDescription, Timing] = {
        "cake:P": Timing(definition="P,p"),
        "cake:S": Timing(definition="S,s"),
    }
    earthmodel: EarthModel = EarthModel()

    _earthmodel: LayeredModel = PrivateAttr()
    _earthmodl_hash: bytes = PrivateAttr()
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

        spatial_bounds = [
            [receiver_depths.min(), receiver_depths.max()],
            [source_depths.min(), source_depths.max()],
            [distances.min(), distances.max()],
        ]
        spatial_tolerance = [octree.size_limit, octree.size_limit, octree.size_limit]
        # TODO: Time tolerance is too hardcoded
        time_tolerance = octree.size_initial / (self.get_vmin() * 5.0)

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
            ftol=np.array(time_tolerance),
            xbounds=np.array(spatial_bounds),
            xtols=spatial_tolerance,
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
