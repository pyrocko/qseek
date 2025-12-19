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

MB = 1024**2

logger = logging.getLogger(__name__)


def weights_gaussian(
    distances: np.ndarray,
    radius: float,
    exponent: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    weights = np.exp(-((distances / radius) ** exponent))
    if normalize:
        weights /= weights.max()
    return weights


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


def weights_gaussian_min_stations(
    distances: np.ndarray,
    radius: float,
    exponent: float = 2.0,
    required_stations: int = 5,
    waterlevel: float = 0.0,
) -> np.ndarray:
    sorted_distances = np.sort(distances, axis=0)
    required_stations = min(required_stations, distances.shape[1] - 1)

    threshold_distances = (
        sorted_distances[:, required_stations][:, np.newaxis]
        if required_stations > 0
        else 0.0
    )

    weights = np.exp(-(((distances - threshold_distances) / radius) ** exponent))
    weights[:, :required_stations] = 1.0
    if waterlevel > 0.0:
        weights = (1 - waterlevel) * weights + waterlevel
    return weights


class DistanceWeights(BaseModel):
    shape: Literal["exponential", "gaussian"] = Field(
        default="exponential",
        description="Shape of the decay function. 'gaussian' or 'exponential'."
        " Default is 'exponential'.",
    )
    radius_meters: PositiveFloat | Literal["mean_interstation"] = Field(
        default="mean_interstation",
        description="Cutoff distance for the spatial decay function in meters."
        " 'mean_interstation' uses the mean interstation distance for the radius."
        " Default is 'mean_interstation'.",
    )
    min_required_stations: PositiveInt = Field(
        default=4,
        description="Minimum number of stations to assign full weight in the"
        " exponential decay function. Default is 5.",
    )
    exponent: float = Field(
        default=2.0,
        description="Exponent of the spatial decay function. For 'gaussian' decay an"
        " exponent of 0.5 is recommended. Default is 2.",
        ge=0.0,
    )
    waterlevel: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Waterlevel for the exponential decay function. Default is 0.0.",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize the weights to the range [0, 1]. Default is True.",
    )

    _node_lut: ArrayLRUCache[bytes] = PrivateAttr()
    _stations: StationList = PrivateAttr()
    _station_coords_ecef: np.ndarray = PrivateAttr()

    def get_distances(self, nodes: Sequence[Node]) -> np.ndarray:
        node_coords = get_node_coordinates(nodes, system="geographic")
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T
        return np.linalg.norm(
            self._station_coords_ecef - node_coords[:, np.newaxis],
            axis=2,
        )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        logger.info("preparing distance weights")

        if self.shape == "gaussian" and self.radius_meters != "mean_interstation":
            logger.warning(
                "gaussian distance weighting uses mean interstation distance as radius"
            )
            self.radius_meters = "mean_interstation"

        if self.radius_meters == "mean_interstation":
            self.radius_meters = stations.mean_interstation_distance()
            logger.info(
                "using mean interstation distance as radius: %g m",
                self.radius_meters,
            )

        self._stations = StationList.from_inventory(stations)
        self._node_lut = ArrayLRUCache(name="distance_weights", short_name="DW")

        sta_coords = get_coordinates(self._stations)
        self._station_coords_ecef = np.array(od.geodetic_to_ecef(*sta_coords.T)).T

        self.fill_lut(nodes=octree.nodes)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling distance weight LUT for %d nodes", len(nodes))
        distances = self.get_distances(nodes)
        node_lut = self._node_lut
        for node, sta_distances in zip(nodes, distances, strict=True):
            node_lut[node.hash()] = sta_distances

    def get_node_weights(self, node: Node, stations: list[Station]) -> np.ndarray:
        try:
            distances = self._node_lut[node.hash()]
        except KeyError:
            self.fill_lut([node])
            return self.get_node_weights(node, stations)

        return weights_gaussian_min_stations(
            distances,
            required_stations=self.min_required_stations,
            radius=self.radius_meters,
            exponent=self.exponent,
            waterlevel=self.waterlevel,
        )

    @alog_call
    async def get_weights(
        self,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        n_nodes = len(nodes)
        station_indices = np.fromiter(
            (self._stations.get_index(sta.nsl) for sta in stations),
            dtype=int,
        )
        n_stations = len(stations)
        distances = np.zeros(shape=(n_nodes, n_stations), dtype=np.float32)

        fill_nodes = []
        node_lut = self._node_lut
        for idx, node in enumerate(nodes):
            try:
                distances[idx] = node_lut[node.hash()][station_indices]
            except KeyError:
                fill_nodes.append(node)

        if fill_nodes:
            self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_weights(nodes, stations)

        return weights_gaussian_min_stations(
            distances,
            required_stations=self.min_required_stations,
            radius=self.radius_meters,
            exponent=self.exponent,
            waterlevel=self.waterlevel,
        )
