from __future__ import annotations

from hashlib import sha1
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from lmdb import Environment
from pydantic import BaseModel, PrivateAttr, root_validator
from pyrocko.cake import LayeredModel, PhaseDef, Ray, m2d, read_nd_model_str

from lassie.models.station import Station, Stations
from lassie.octree import Node, Octree
from lassie.tracers.base import RayTracer

if TYPE_CHECKING:
    from lmdb import _Database


class Timing(BaseModel):
    definition: str

    def as_phase_def(self) -> PhaseDef:
        return PhaseDef(definition=self.definition)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[str, Timing] = {
        "cake:P": Timing(definition="pP"),
        "cake:S": Timing(definition="sS"),
    }
    earthmodel: str
    cache: Path | None = None

    _earthmodel: LayeredModel = PrivateAttr()
    _earthmodel_hash: str = PrivateAttr()
    _lmdb: Environment = PrivateAttr()

    @root_validator
    def _read_earthmodel(cls, values):  # noqa N805
        try:
            LayeredModel.from_scanlines(read_nd_model_str(values.get("earthmodel")))
        except Exception as exc:
            raise ValueError("bad earthmodel") from exc
        cache = values["cache"]
        if not cache:
            cache = Path("/tmp/test-lmdb")
        if not cache.exists():
            cache.mkdir(parents=True)

        values["cache"] = cache
        return values

    def __init__(self, **data):
        super().__init__(**data)

        self._earthmodel = LayeredModel.from_scanlines(
            read_nd_model_str(self.earthmodel)
        )
        data = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            self._earthmodel.profile(param).dump(data)
        self._earthmodel_hash = sha1(data.getvalue()).hexdigest()

        self._lmdb = Environment(str(self.cache), sync=False)

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.timings.keys())

    def get_traveltime(self, phase: str, node: Node, station: Station) -> float:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")
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
        distances = octree.distances_stations(stations)
        return np.array(
            [
                self._calculate_traveltimes(phase, node, distance)
                for node, distance in zip(octree, distances)
            ]
        )

    def _get_database(self, phase: str) -> _Database:
        return self._lmdb.open_db(hash((phase, self._earthmodel_hash)))

    def _retrieve_traveltimes(
        self, phase: str, node: Node, distances: np.ndarray
    ) -> np.ndarray:
        database = self._get_database(phase)
        key = hash((node, hash(distances.tobytes())))

        with self._lmdb.begin(write=False) as txn:
            traveltimes = txn.get(key, db=database)
            if traveltimes:
                return np.frombuffer(traveltimes, dtype=np.float32)

        traveltimes = self._calculate_traveltimes(phase, node, distances)
        with self._lmdb.begin(write=True) as txn:
            txn.put(key, traveltimes.tobytes(), db=database)
        return traveltimes

    def _calculate_traveltimes(
        self, phase: str, node: Node, distances: np.ndarray
    ) -> np.ndarray:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")

        print("Calculating")
        rays: list[Ray] = self._earthmodel.arrivals(
            distances=distances * m2d,
            phases=self.timings[phase].as_phase_def(),
            zstart=node.depth,
        )
        return np.array([ray.t for ray in rays]).astype(np.float32)
