from __future__ import annotations

import logging
import struct
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from lmdb import Environment
from pydantic import BaseModel, PrivateAttr, root_validator
from pyrocko.cake import LayeredModel, PhaseDef, Ray, m2d, read_nd_model_str

from lassie.models.station import Station, Stations
from lassie.octree import Node, Octree
from lassie.tracers.base import RayTracer

if TYPE_CHECKING:
    from lmdb import _Database

MAX_DBS = 16
logger = logging.getLogger(__name__)


def hash_to_bytes(hash: int) -> bytes:
    return struct.pack("q", hash)


class Timing(BaseModel):
    definition: str

    def as_phase_def(self) -> PhaseDef:
        return PhaseDef(definition=self.definition)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[str, Timing] = {
        "cake:P": Timing(definition="P"),
        "cake:S": Timing(definition="S"),
    }
    earthmodel: str
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
        self._init_lmdb()

    def clear_cache(self) -> None:
        """Clear the lmdb cache."""
        self._lmdb.close()
        self.cache.unlink()
        self._init_lmdb()

    def _init_lmdb(self) -> None:
        logger.info("using lmdb cache %s", self.cache)
        self._lmdb = Environment(str(self.cache), max_dbs=MAX_DBS, create=True)
        with self._lmdb.begin(write=True) as txn:
            cursor = txn.cursor()
            for idx_db, db_hash in enumerate(cursor.iterprev(values=False)):
                if idx_db >= MAX_DBS - 1:
                    logger.warning("deleting old traveltimes %s", db_hash)
                    cursor.pop(db_hash)

    def cache_info(self) -> dict[str, Any]:
        return self._lmdb.info()

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.timings.keys())

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

        distances = octree.distances_stations(stations)
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

        with self._lmdb.begin(write=False) as txn:
            traveltimes = txn.get(node_key, db=database)
            if traveltimes:
                self.cache_hits += 1
                return np.frombuffer(traveltimes, dtype=np.float32)

        traveltimes = self._calculate_node_traveltimes(timing, node, distances)
        with self._lmdb.begin(write=True) as txn:
            txn.put(node_key, traveltimes.tobytes(), db=database)
        return traveltimes

    def _calculate_node_traveltimes(
        self, timing: PhaseDef, node: Node, distances: np.ndarray
    ) -> np.ndarray:
        logger.debug("calculating traveltimes")
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
