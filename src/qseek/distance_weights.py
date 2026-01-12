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
    taper_distance: float = 0.0,
    n_stations_plateau: int = 4,
    waterlevel: float = 0.0,
) -> np.ndarray:
    if n_stations_plateau < 1:
        raise ValueError("required_stations must be at least 1")

    n_stations_plateau = min(n_stations_plateau, distances.shape[1])
    sorted_distances = np.sort(distances, axis=1)
    threshold_distance = sorted_distances[:, n_stations_plateau - 1, np.newaxis]

    # k = 9.19 / distance_taper  # 1-99% range
    k = 7.84 / taper_distance  # 2-98% range
    return 1 - (
        (1 - waterlevel)
        / (1 + np.exp(-k * (distances - (threshold_distance + taper_distance / 2))))
    )


def weights_gaussian(
    distances: np.ndarray,
    n_stations_plateau: int = 4,
    n_stations_taper: int = 5,
    taper_distance: float = 1000.0,
    waterlevel: float = 0.0,
) -> np.ndarray:
    if n_stations_plateau < 1:
        raise ValueError("required_stations must be at least 1")
    if n_stations_taper < 0:
        raise ValueError("taper_stations must be at least 0")
    if taper_distance < 0.0:
        raise ValueError("distance_taper must be positive")

    n_stations_plateau = min(n_stations_plateau, distances.shape[1])
    sorted_distances = np.sort(distances, axis=1)
    threshold_distance = sorted_distances[:, n_stations_plateau - 1, np.newaxis]

    if n_stations_taper > 0:
        n_stations_taper = min(
            n_stations_taper + n_stations_plateau, distances.shape[1]
        )
        taper_distance = sorted_distances[:, n_stations_taper - 1, np.newaxis]
        taper_distance += threshold_distance
    else:
        taper_distance = (threshold_distance + taper_distance) / 2.355

    # Full width at half maximum (FWHM) to standard deviation conversion:
    # FWHM = 2.355 * sigma
    weights = np.exp(
        -(((distances - threshold_distance) ** 2) / (2 * (taper_distance) ** 2))
    )
    weights[distances <= threshold_distance] = 1.0
    if waterlevel > 0.0:
        weights = (1 - waterlevel) * weights + waterlevel
    return weights


class DistanceWeights(BaseModel):
    taper_distance: PositiveFloat | Literal["mean_interstation"] = Field(
        default="mean_interstation",
        description="Taper distance for the Gaussian weighting function"
        " in meters. 'mean_interstation' uses twice the mean interstation distance for"
        " the radius. Default is 'mean_interstation'.",
    )
    n_stations_taper: int = Field(
        default=8,
        ge=0,
        description="Number of stations to calculate the taper distance in the"
        " spatial weighting function. Default is 5.",
    )
    n_stations_plateau: PositiveInt = Field(
        default=4,
        description="The number of closest stations to retain with full weight (1.0). "
        "These stations form the 'plateau' of the spatial weighting function, "
        "ensuring the core aperture is never down-weighted. Default is 4.",
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
        if self.taper_distance == "mean_interstation":
            self.taper_distance = 2 * stations.mean_interstation_distance()
            logger.info(
                "using 2x mean interstation distance as distance taper: %g m",
                self.taper_distance,
            )
        logger.info(
            "distance weighting uses %d closest stations and a taper of %g m",
            self.n_stations_plateau,
            self.taper_distance,
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
            return weights_gaussian(
                np.array(distances),
                n_stations_plateau=self.n_stations_plateau,
                n_stations_taper=self.n_stations_taper,
                taper_distance=self.taper_distance,
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
