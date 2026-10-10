"""Station weights decide how much each station contributes to the stack of a node."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Annotated, Literal, Sequence, Union

import numpy as np
import pyrocko.orthodrome as od
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr

from qseek.base import Model
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

PLATEAU_WEIGHT_DESCRIPTION = (
    "Number of independent stations of a node that get full weight: the closest"
    " stations up to this cumulative station weight. A station in a dense cluster"
    " counts less than an isolated station, see `station_density_weights`."
)


def weights_gaussian(
    distances: np.ndarray,
    distance_taper: float,
    required_stations: int = 4,
    waterlevel: float = 0.0,
) -> np.ndarray:
    """Gaussian distance weights with a plateau of the closest stations.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        distance_taper: Full width at half maximum of the Gaussian taper in meters.
        required_stations: Number of closest stations with full weight.
        waterlevel: Minimum weight of distant stations, as a fraction of the full
            weight.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
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


def interstation_distances(locations: Sequence[Location]) -> np.ndarray:
    """Calculate the distances between stations.

    Args:
        locations: Station locations.

    Returns:
        Array of shape (n_stations, n_stations) with the interstation distances in
            meters. The diagonal is NaN.
    """
    coords = get_coordinates(locations, system="geographic")
    coords_ecef = np.array(od.geodetic_to_ecef(*coords.T)).T
    distances = np.linalg.norm(coords_ecef - coords_ecef[:, np.newaxis], axis=2)
    np.fill_diagonal(distances, np.nan)
    return distances


def nearest_neighbor_distance(
    distances: np.ndarray,
    min_distance: float = 1.0,
) -> float:
    """Calculate the median distance between neighboring station sites.

    Co-located stations, e.g. a broadband and a strong-motion sensor of one site,
    are one site: distances up to `min_distance` are ignored.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters.
        min_distance: Stations closer than this distance in meters are co-located.

    Returns:
        Median nearest neighbor distance in meters, NaN with less than two sites.
    """
    distances = np.where(np.atleast_2d(distances) > min_distance, distances, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(np.nanmin(distances, axis=1)))


def station_density(distances: np.ndarray, radius: float = 0.0) -> np.ndarray:
    """Calculate the station density from a Gaussian kernel.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.
        radius: Standard deviation of the kernel in meters. If 0.0, the median
            nearest neighbor distance between sites.

    Returns:
        Array of shape (n_stations,) with the density at each station, 1.0 for a
            station without neighbors.
    """
    if radius <= 0.0:
        radius = nearest_neighbor_distance(distances)
    if not np.isfinite(radius):
        return np.ones(distances.shape[0])
    kernel = np.exp(-(distances**2) / (2 * radius**2))
    return np.nansum(kernel, axis=1) + 1.0


def station_density_weights(distances: np.ndarray) -> np.ndarray:
    """Calculate the weight of each station from the station density.

    Stations in dense clusters carry less independent information than isolated
    stations. The station with the lowest density gets weight 1.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.

    Returns:
        Array of shape (n_stations,) with weights between 0 and 1.
    """
    density = station_density(distances)
    return 1.0 - (density - density.min()) / density.max()


def cumulative_weight_distance(
    distances: np.ndarray,
    station_weights: np.ndarray,
    cumulative_weight: float,
) -> np.ndarray:
    """Distance at which the cumulative weight of the closest stations is reached.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        station_weights: Array of shape (n_stations,) with the station weights.
        cumulative_weight: Cumulative station weight to reach.

    Returns:
        Array of shape (n_nodes, 1) with the distance of the station at which the
            cumulative weight is reached, or of the most distant station if the
            total station weight is lower.
    """
    order = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, order, axis=1)
    cumulative = np.cumsum(station_weights[order], axis=1)

    idx = np.argmax(cumulative >= cumulative_weight, axis=1)
    idx[cumulative[:, -1] < cumulative_weight] = distances.shape[1] - 1
    return sorted_distances[np.arange(distances.shape[0]), idx, np.newaxis]


def weights_density_gaussian(
    distances: np.ndarray,
    station_weights: np.ndarray,
    plateau_weight: float = 4.0,
    taper_weight: float = 12.0,
) -> np.ndarray:
    """Gaussian taper with a plateau of the closest independent stations.

    The plateau ends where the cumulative station weight reaches `plateau_weight`.
    The Gaussian taper starts there; its standard deviation is half the distance at
    which the cumulative station weight reaches `taper_weight`.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        station_weights: Array of shape (n_stations,) with station weights.
        plateau_weight: Cumulative station weight of the plateau.
        taper_weight: Cumulative station weight that sets the taper width.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    plateau = cumulative_weight_distance(distances, station_weights, plateau_weight)
    taper = cumulative_weight_distance(distances, station_weights, taper_weight)
    sigma = taper / 2

    weights = np.exp(-((distances - plateau) ** 2) / (2 * sigma**2))
    weights[distances <= plateau] = 1.0
    return weights


def weights_log_logistic(
    distances: np.ndarray,
    plateau_distances: np.ndarray,
    taper_scale: float = 2.2,
    taper_exponent: float = 4.0,
) -> np.ndarray:
    """Log-logistic taper relative to the plateau distance of each node.

    The weight is `1 / (1 + (d / (taper_scale * d_plateau)) ** taper_exponent)`:
    0.5 at `taper_scale` times the plateau distance, independent of the scale of
    the network.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        plateau_distances: Array of shape (n_nodes, 1) with the plateau distance of
            each node in meters.
        taper_scale: Distance of half weight, in units of the plateau distance.
        taper_exponent: Exponent of the taper, how fast the weights decay.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    half_weight = taper_scale * np.maximum(plateau_distances, 1.0)
    return 1.0 / (1.0 + (distances / half_weight) ** taper_exponent)


class StationWeights(Model):
    """Base class of the station weights.

    Station weights are calculated for every pair of node and station from their
    distance, and normalized per node in the stack. Subclasses set the weights with
    `calculate_weights`.
    """

    weights: Literal["StationWeights"] = "StationWeights"

    waterlevel: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum weight of distant stations, as a fraction of the full"
        " weight. With `0.0`, distant stations do not contribute.",
    )

    _node_lut: ArrayLRUCache[bytes] = PrivateAttr()
    _stations: StationList = PrivateAttr()
    _station_coords_ecef: np.ndarray = PrivateAttr()
    _interstation_distances: np.ndarray | None = PrivateAttr(None)
    _density_weights: dict[bytes, np.ndarray] = PrivateAttr(default_factory=dict)

    @classmethod
    def get_subclasses(cls) -> tuple[type[StationWeights], ...]:
        """Get the subclasses of this class.

        Returns:
            tuple[type[StationWeights], ...]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    def get_distances(self, nodes: Sequence[Node]) -> np.ndarray:
        """Get the distances between nodes and all stations.

        Args:
            nodes: Nodes of the octree.

        Returns:
            Array of shape (n_nodes, n_stations) with distances in meters.
        """
        node_coords = get_node_coordinates(nodes, system="geographic")
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T), dtype=np.float32).T

        return np.linalg.norm(
            self._station_coords_ecef - node_coords[:, np.newaxis],
            axis=2,
        )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        """Prepare the station weights for the stations and the octree.

        Args:
            stations: Stations of the search.
            octree: Octree of the search.
        """
        self._stations = StationList.from_inventory(stations)
        self._node_lut = ArrayLRUCache(name="station_weights", short_name="SW")
        self._interstation_distances = None
        self._density_weights = {}

        sta_coords = get_coordinates(self._stations)
        self._station_coords_ecef = np.array(
            od.geodetic_to_ecef(*sta_coords.T), dtype=np.float32
        ).T

        self.fill_lut(nodes=octree.nodes)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        """Calculate the node-station distances of nodes for the lookup table.

        Args:
            nodes: Nodes of the octree.
        """
        logger.debug("filling station weight LUT for %d nodes", len(nodes))

        distances = self.get_distances(nodes)
        node_lut = self._node_lut
        for node, sta_distances in zip(nodes, distances, strict=True):
            node_lut[node.hash] = sta_distances

    def get_density_weights(self, station_indices: np.ndarray) -> np.ndarray:
        """Get the station density weights of a set of stations.

        The weights depend on the stations that are available, they are cached for
        each set of stations.

        Args:
            station_indices: Indices of the available stations.

        Returns:
            Array of shape (n_stations,) with the station density weights.
        """
        key = station_indices.tobytes()
        if key not in self._density_weights:
            if self._interstation_distances is None:
                self._interstation_distances = interstation_distances(
                    list(self._stations)
                )
            distances = self._interstation_distances[station_indices][
                :, station_indices
            ]
            self._density_weights[key] = station_density_weights(distances)
        return self._density_weights[key]

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
    ) -> np.ndarray:
        """Calculate the weights from the node-station distances.

        Args:
            distances: Array of shape (n_nodes, n_stations) with node-station
                distances in meters.
            station_indices: Indices of the stations in the station list.

        Returns:
            Array of shape (n_nodes, n_stations) with weights between 0 and 1,
                before the waterlevel.
        """
        raise NotImplementedError

    @alog_call
    async def get_weights(
        self,
        nodes: Sequence[Node],
        stations: Sequence[Station],
    ) -> np.ndarray:
        """Get the weights of the stations for nodes.

        Args:
            nodes: Nodes of the octree.
            stations: Available stations.

        Returns:
            Array of shape (n_nodes, n_stations) with weights between 0 and 1.
        """
        node_lut = self._node_lut
        station_indices = self._stations.get_indices(stations)

        try:
            distances = [node_lut[node.hash][station_indices] for node in nodes]
        except KeyError:
            fill_nodes = [node for node in nodes if node.hash not in node_lut]

            self.fill_lut(fill_nodes)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                node_lut.fill_level() * 100,
                node_lut.hit_rate() * 100,
            )
            return await self.get_weights(nodes, stations)

        weights = self.calculate_weights(np.array(distances), station_indices)
        if self.waterlevel > 0.0:
            weights = (1 - self.waterlevel) * weights + self.waterlevel
        return weights


class DistanceWeights(StationWeights):
    """The closest stations get full weight, more distant stations a Gaussian taper.

    Close stations constrain the location of an event best. The taper width is
    absolute, by default twice the mean interstation distance.
    """

    weights: Literal["DistanceWeights"] = "DistanceWeights"

    distance_taper: PositiveFloat | Literal["mean_interstation"] = Field(
        default="mean_interstation",
        description=(
            "Distance in meters over which the weight of distant stations decays with a"
            ' Gaussian function. `"mean_interstation"` uses twice the mean interstation'
            " distance of the network."
        ),
    )
    required_closest_stations: PositiveInt = Field(
        default=4,
        description=(
            "Number of closest stations of a node that get full weight. Only more "
            "distant stations are tapered, so that the closest stations contribute "
            "equally and the most to the detection and localization."
        ),
    )

    _distance_taper: float = PrivateAttr()

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        if self.distance_taper == "mean_interstation":
            self.distance_taper = 2 * stations.mean_interstation_distance()
            logger.info(
                "using 2x mean interstation distance as distance taper: %g m",
                self.distance_taper,
            )
        self._distance_taper = self.distance_taper
        logger.info(
            "distance weighting uses %d closest stations and a taper of %g m",
            self.required_closest_stations,
            self._distance_taper,
        )
        super().prepare(stations, octree)

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
    ) -> np.ndarray:
        return weights_gaussian(
            distances,
            distance_taper=self._distance_taper,
            required_stations=self.required_closest_stations,
        )


class StationDensityWeights(StationWeights):
    """The closest independent stations get full weight, then a Gaussian taper.

    Every station carries a weight from the station density: stations in dense
    clusters count less than isolated ones. The plateau of full weight and the
    taper are set in units of this cumulative station weight, so they adapt to the
    station spacing around each node.
    """

    weights: Literal["StationDensityWeights"] = "StationDensityWeights"

    plateau_weight: PositiveFloat = Field(
        default=4.0,
        description=PLATEAU_WEIGHT_DESCRIPTION,
    )
    taper_weight: PositiveFloat = Field(
        default=12.0,
        description="Cumulative station weight that sets the width of the Gaussian"
        " taper: its standard deviation is half the distance at which the closest"
        " stations reach this weight. If the network has less station weight, the"
        " most distant station sets the width.",
    )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        super().prepare(stations, octree)
        all_stations = np.arange(len(self._stations))
        total_weight = float(self.get_density_weights(all_stations).sum())
        logger.info(
            "station density weights: total weight %.1f of %d stations",
            total_weight,
            len(self._stations),
        )
        if total_weight < self.taper_weight:
            logger.info(
                "taper_weight %g exceeds the total station weight %.1f,"
                " the most distant station sets the taper width",
                self.taper_weight,
                total_weight,
            )

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
    ) -> np.ndarray:
        return weights_density_gaussian(
            distances,
            self.get_density_weights(station_indices),
            plateau_weight=self.plateau_weight,
            taper_weight=self.taper_weight,
        )


class LogLogisticWeights(StationWeights):
    """The closest independent stations set the plateau, then a log-logistic taper.

    The plateau distance of a node is the distance at which the closest stations
    reach the cumulative station weight `plateau_weight`, as for
    `StationDensityWeights`. The weights decay with the distance in units of the
    plateau distance, `1 / (1 + (d / (taper_scale * d_plateau)) ** taper_exponent)`,
    like the confidence of the phase picks of small events.
    """

    weights: Literal["LogLogisticWeights"] = "LogLogisticWeights"

    plateau_weight: PositiveFloat = Field(
        default=4.0,
        description=PLATEAU_WEIGHT_DESCRIPTION,
    )
    taper_scale: PositiveFloat = Field(
        default=2.2,
        description="Distance at which the weight is 0.5, in units of the plateau"
        " distance of the node.",
    )
    taper_exponent: PositiveFloat = Field(
        default=4.0,
        description="Exponent of the log-logistic taper. Higher values decay"
        " faster beyond `taper_scale` times the plateau distance.",
    )

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
    ) -> np.ndarray:
        plateau = cumulative_weight_distance(
            distances,
            self.get_density_weights(station_indices),
            self.plateau_weight,
        )
        return weights_log_logistic(
            distances,
            plateau,
            taper_scale=self.taper_scale,
            taper_exponent=self.taper_exponent,
        )


# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type StationWeightsType = StationWeights
else:
    type StationWeightsType = Annotated[
        Union[StationWeights.get_subclasses()],
        Field(discriminator="weights"),
    ]
