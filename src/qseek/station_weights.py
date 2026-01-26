from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pyrocko.orthodrome as od
from pydantic import BaseModel, Field, PositiveFloat, PrivateAttr

from qseek.cache_lru import ArrayLRUCache
from qseek.models.location import get_coordinates
from qseek.models.station import StationList
from qseek.octree import get_node_coordinates
from qseek.utils import alog_call

if TYPE_CHECKING:
    from qseek.models.location import Location
    from qseek.models.station import Station, StationInventory
    from qseek.octree import Node, Octree


logger = logging.getLogger(__name__)

KM = 1e3


def _get_interstation_distances(locations: Sequence[Location]) -> np.array:
    """Calculate interstation distances.

    Args:
        locations: List of station locations.

    Returns:
        Array of shape (n_stations, n_stations) with interstation distances in meters
            The diagonal is filled with NaNs.
    """
    coords = get_coordinates(locations, system="geographic")
    coords_ecef = np.array(od.geodetic_to_ecef(*coords.T)).T
    distances = np.linalg.norm(coords_ecef - coords_ecef[:, np.newaxis], axis=2)
    np.fill_diagonal(distances, np.nan)
    return distances.astype(np.float32)


def _nn_median(distances: np.ndarray) -> float:
    """Calculate the median nearest neighbor distance between stations.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation distances
            in meters.

    Returns:
        Median nearest neighbor distance in meters.
    """
    distances = np.atleast_2d(distances)
    return np.nanmedian(np.nanmin(distances, axis=1))


def _station_density(
    interstation_distances: np.ndarray,
    radius: float = 0.0,
) -> np.ndarray:
    """Calculate station density from Gaussian kernel densities.

    Args:
        interstation_distances: Array of shape (n_stations, n_stations) with
            interstation distances in meters.
        radius: Kernel radius in meters. If 0.0, the median nearest neighbor distance
            is used. Default is 0.0.

    Returns:
        Array of shape (n_stations,) with station densities.
    """
    if radius <= 0.0:
        radius = _nn_median(interstation_distances)
    # radius: sigma of the Gaussian kernel
    kde = np.nansum(np.exp(-((interstation_distances**2) / (2 * radius**2))), axis=1)
    kde += 1.0  # add station self-contribution
    return kde


def _station_weights(
    insterstation_distances: np.ndarray, radius: float = 0.0
) -> np.ndarray:
    """Calculate station weights from station densities.

    Args:
        insterstation_distances: Array of shape (n_stations, n_stations) with
            interstation distances in meters.
        radius: Kernel radius in meters. If 0.0, the median nearest neighbor distance
            is used. Default is 0.0.

    Returns:
        Array of shape (n_stations,) with station weights between 0 and 1.
    """
    station_densities = _station_density(insterstation_distances, radius=radius)
    return 1 - (station_densities - station_densities.min()) / station_densities.max()


def distance_weights(
    distances: np.ndarray,
    station_weights: np.ndarray,
    weight_plateau: float = 4.0,
    weight_taper: float = 12.0,
) -> np.ndarray:
    """Calculate distance weights with plateau and taper based on station weights.

    The stations weights act as an information content measure. The plateau and taper
    distances are defined based on cumulative station weights.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        station_weights: Array of shape (n_stations,) with station weights between
            0 and 1.
        weight_plateau: Cumulative station weight theshold to define the plateau
            distance. Default is 4.0.
        weight_taper: Cumulative Station weight threshold to define the taper
            distance. Default is 12.0.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    n_nodes = distances.shape[0]

    distance_sort = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, distance_sort, axis=1)
    sorted_station_weights = station_weights[distance_sort]

    cum_station_weights = np.cumsum(sorted_station_weights, axis=1)

    # First index where cumulative weights exceed plateau and taper weights
    idxs_plateau = np.argmax(cum_station_weights >= weight_plateau, axis=1)
    idxs_taper = np.argmax(cum_station_weights >= weight_taper, axis=1)

    # If total cumulative weight is less than plateau/taper, set to last index
    idx_last_station = distances.shape[1] - 1
    idxs_plateau[cum_station_weights[:, -1] < weight_plateau] = idx_last_station
    idxs_taper[cum_station_weights[:, -1] < weight_taper] = idx_last_station

    plateau_distances = sorted_distances[np.arange(n_nodes), idxs_plateau, np.newaxis]
    taper_distance = sorted_distances[np.arange(n_nodes), idxs_taper, np.newaxis]
    # We use half sigma here, more weight will estimate better geometry of the
    # station distribution
    taper_sigma = taper_distance / 2

    distance_weights = np.exp(
        -(((distances - plateau_distances) ** 2) / (2 * taper_sigma**2))
    )
    distance_weights[distances <= plateau_distances] = 1.0
    return distance_weights


def weights_gaussian(
    distances: np.ndarray,
    n_stations_plateau: int = 4,
    n_stations_taper_distance: int = 5,
    taper_distance: float = 1000.0,
    waterlevel: float = 0.0,
) -> np.ndarray:
    """Gaussian distance weights with plateau and taper.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        n_stations_plateau: Number of closest stations to retain with full weight (1.0).
            Default is 4.
        n_stations_taper_distance: Number of stations to define the taper width.
            If 0, taper_distance is used instead. Default is 5.
        taper_distance: Taper distance in meters if n_stations_taper_distance is 0.
            The taper distance is given as full width at half maximum (FWHM).
            Default is 1000.0.
        waterlevel: Stations outside the taper are lifted by this fraction.
            Default is 0.0.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    if n_stations_plateau < 1:
        raise ValueError("required_stations must be at least 1")
    if n_stations_taper_distance < 0:
        raise ValueError("taper_stations must be at least 0")
    if taper_distance < 0.0:
        raise ValueError("distance_taper must be positive")

    n_stations_plateau = min(n_stations_plateau, distances.shape[1])
    sorted_distances = np.sort(distances, axis=1)
    threshold_distance = sorted_distances[:, n_stations_plateau - 1, np.newaxis]

    if n_stations_taper_distance > 0:
        n_stations_taper = n_stations_taper_distance + n_stations_plateau
        idx_stations_taper = min(n_stations_taper, distances.shape[1])

        sigma_distance = sorted_distances[:, idx_stations_taper - 1, np.newaxis]
        # Linearly interpolate taper distance if station count is less
        # than requested stations for taper to broaden the taper.
        if idx_stations_taper < n_stations_taper:
            sigma_distance = (sigma_distance / idx_stations_taper) * n_stations_taper
    else:
        # Full width at half maximum (FWHM) to standard deviation conversion:
        # FWHM = 2.355 * sigma
        sigma_distance = taper_distance / 2.355

    weights = np.exp(
        -(((distances - threshold_distance) ** 2) / (2 * sigma_distance**2))
    )
    weights[distances <= threshold_distance] = 1.0

    if waterlevel > 0.0:
        weights = (1.0 - waterlevel) * weights + waterlevel
    return weights


class StationWeights(BaseModel):
    plateau_weight: PositiveFloat = Field(
        default=4.0,
        description="The cumulative station weight required to define the"
        "'Core Aperture' (Plateau). A value of 4.0 ensures the location is constrained"
        "by the equivalent of 4 independent, high-quality stations.",
    )
    taper_weight: PositiveFloat = Field(
        default=12.0,
        description="The cumulative station weight threshold that defines where the "
        "Gaussian taper reaches its effective limit. This value determines how many "
        "equivalent stations contribute to the location estimate beyond the core "
        "aperture. Higher values (e.g., 20.0) include more distant stations with "
        "gradually decreasing weights, while lower values (e.g., 8.0) create a "
        "more localized influence zone. Default is 12.0.",
    )

    _node_distance_lut: ArrayLRUCache[bytes] = PrivateAttr()
    _stations: StationList = PrivateAttr()
    _station_coords_ecef: np.ndarray = PrivateAttr()
    _interstation_distances: np.ndarray = PrivateAttr()

    def get_distances(self, nodes: Sequence[Node]) -> np.ndarray:
        node_coords = get_node_coordinates(nodes, system="geographic")
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T), dtype=np.float32).T
        return np.linalg.norm(
            self._station_coords_ecef - node_coords[:, np.newaxis],
            axis=2,
        )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        self._stations = StationList.from_inventory(stations)
        self._node_distance_lut = ArrayLRUCache(
            name="distance_weights", short_name="DW"
        )

        sta_coords = get_coordinates(self._stations)
        self._station_coords_ecef = np.array(
            od.geodetic_to_ecef(*sta_coords.T),
            dtype=np.float32,
        ).T
        self._interstation_distances = _get_interstation_distances(list(self._stations))

        logger.info("calculating overall station weights")
        station_weights = _station_weights(self._interstation_distances)
        for sta, weight in zip(stations, station_weights, strict=True):
            sta.set_apparent_weight(float(weight))

        self.fill_lut(nodes=octree.nodes)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling distance weight LUT for %d nodes", len(nodes))

        distances = self.get_distances(nodes)
        node_lut = self._node_distance_lut
        for node, sta_distances in zip(nodes, distances, strict=True):
            node_lut[node.hash] = sta_distances

    @alog_call
    async def get_weights(
        self,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        node_lut = self._node_distance_lut
        station_idxs = self._stations.get_indexes(stations)

        try:
            distances = [node_lut[node.hash][station_idxs] for node in nodes]

            weights_stations = _station_weights(
                self._interstation_distances[station_idxs][:, station_idxs]
            )
            weights_distance = distance_weights(
                distances=np.asarray(distances),
                station_weights=weights_stations,
                weight_plateau=self.plateau_weight,
                weight_taper=self.taper_weight,
            )
            weights = weights_distance * weights_stations
            return weights / weights.max(axis=1, keepdims=True)

        except KeyError:
            fill_nodes = [node for node in nodes if node.hash not in node_lut]

            self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_weights(nodes, stations)
