from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pyrocko.orthodrome as od
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, PrivateAttr

from qseek.cache_lru import ArrayLRUCache
from qseek.models.location import get_coordinates
from qseek.models.station import StationList
from qseek.octree import get_node_coordinates
from qseek.utils import alog_call

if TYPE_CHECKING:
    from qseek.models.station import Station, StationInventory
    from qseek.octree import Node, Octree


logger = logging.getLogger(__name__)


def weights_exponential(
    distances: np.ndarray,
    radius: float,
    exponent: float = 2.0,
    waterlevel: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    weights = (1 - waterlevel) / (1 + (distances / radius) ** exponent) + waterlevel
    if normalize:
        weights /= weights.max()
    return weights


def weights_logistic(
    distances: np.ndarray,
    distance_taper: float,
    required_stations: int = 4,
    waterlevel: float = 0.0,
) -> np.ndarray:
    if required_stations < 1:
        raise ValueError("required_stations must be at least 1")

    required_stations = min(required_stations, distances.shape[1])
    sorted_distances = np.sort(distances, axis=1)
    threshold_distance = sorted_distances[:, required_stations - 1, np.newaxis]

    # k = 9.19 / distance_taper  # 1-99% range
    k = 7.84 / distance_taper  # 2-98% range
    return 1 - (
        (1 - waterlevel)
        / (1 + np.exp(-k * (distances - (threshold_distance + distance_taper / 2))))
    )


def weights_gaussian(
    distances: np.ndarray,
    distance_taper: float,
    required_stations: int = 4,
    waterlevel: float = 0.0,
) -> np.ndarray:
    if required_stations < 1:
        raise ValueError("required_stations must be at least 1")

    required_stations = min(required_stations, distances.shape[1])
    sorted_distances = np.sort(distances, axis=1)
    threshold_distance = sorted_distances[:, required_stations - 1, np.newaxis]

    # Full width at half maximum (FWHM) to standard deviation conversion:
    # FWHM = 2.355 * sigma
    weights = np.exp(
        -(((distances - threshold_distance) ** 2) / (2 * (distance_taper / 2.355) ** 2))
    )
    weights[distances <= threshold_distance] = 1.0
    if waterlevel > 0.0:
        weights = (1 - waterlevel) * weights + waterlevel
    return weights


class DistanceWeights(BaseModel):
    distance_taper: PositiveFloat | Literal["mean_interstation"] = Field(
        default="mean_interstation",
        description="Taper distance for the Gaussian weighting function"
        " in meters. 'mean_interstation' uses twice the mean interstation distance for"
        " the radius. Default is 'mean_interstation'.",
    )
    required_closest_stations: PositiveInt = Field(
        default=4,
        description="Number of stations to assign full weight in the"
        " spatial weighting function, only more distant stations are tapered with a"
        " Gaussian decay. This ensures that the closest _N_ stations have an equal and"
        " the highest contribution to the detection and localization. Default is 4.",
    )
    waterlevel: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Stations outside the taper are lifted by this fraction. "
        "Default is 0.0.",
    )

    _node_lut: ArrayLRUCache[bytes] = PrivateAttr()
    _stations: StationList = PrivateAttr()
    _station_coords_ecef: np.ndarray = PrivateAttr()

    def get_distances(self, nodes: Sequence[Node]) -> np.ndarray:
        node_coords = get_node_coordinates(nodes, system="geographic")
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T), dtype=np.float32).T

        return np.linalg.norm(
            self._station_coords_ecef - node_coords[:, np.newaxis],
            axis=2,
        )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        if self.distance_taper == "mean_interstation":
            self.distance_taper = 2 * stations.mean_interstation_distance()
            logger.info(
                "using 2x mean interstation distance as distance taper: %g m",
                self.distance_taper,
            )
        logger.info(
            "distance weighting uses %d closest stations and a taper of %g m",
            self.required_closest_stations,
            self.distance_taper,
        )

        self._stations = StationList.from_inventory(stations)
        self._node_lut = ArrayLRUCache(name="distance_weights", short_name="DW")

        sta_coords = get_coordinates(self._stations)
        self._station_coords_ecef = np.array(
            od.geodetic_to_ecef(*sta_coords.T), dtype=np.float32
        ).T

        self.fill_lut(nodes=octree.nodes)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling distance weight LUT for %d nodes", len(nodes))

        distances = self.get_distances(nodes)
        node_lut = self._node_lut
        for node, sta_distances in zip(nodes, distances, strict=True):
            node_lut[node.hash] = sta_distances

    def get_node_weights(self, node: Node, stations: list[Station]) -> np.ndarray:
        try:
            distances = self._node_lut[node.hash]
        except KeyError:
            self.fill_lut([node])
            return self.get_node_weights(node, stations)

        return weights_gaussian(
            distances,
            required_stations=self.required_closest_stations,
            distance_taper=self.distance_taper,
            waterlevel=self.waterlevel,
        )

    @alog_call
    async def get_weights(
        self,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        node_lut = self._node_lut
        station_indices = self._stations.get_indices(stations)

        try:
            distances = [node_lut[node.hash][station_indices] for node in nodes]
            return weights_logistic(
                np.array(distances),
                required_stations=self.required_closest_stations,
                distance_taper=self.distance_taper,
                waterlevel=self.waterlevel,
            )
        except KeyError:
            fill_nodes = [node for node in nodes if node.hash not in node_lut]

            self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_weights(nodes, stations)
