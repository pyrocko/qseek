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

# Sensors closer than this share a site, e.g. a broadband and a strong-motion sensor
COLOCATED_DISTANCE = 50.0
# Taper width of the distance weights in units of the nearest neighbor distance
NEAREST_NEIGHBOR_TAPER = 6.0

PLATEAU_STATIONS_DESCRIPTION = (
    "Number of independent stations of a node that get full weight. The closest"
    " stations get full weight until they add up to this number of independent"
    " stations. An isolated station counts as one independent station, a station in"
    " a dense cluster as a fraction of one, co-located sensors as one together."
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


def colocated_sensors(distances: np.ndarray) -> np.ndarray:
    """Count the sensors of each station's site.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.

    Returns:
        Array of shape (n_stations,) with the number of sensors within
            `COLOCATED_DISTANCE`, including the station itself.
    """
    return 1 + np.sum(distances <= COLOCATED_DISTANCE, axis=1)


def nearest_neighbor_distance(distances: np.ndarray) -> float:
    """Calculate the median distance between neighboring station sites.

    Co-located sensors, closer than `COLOCATED_DISTANCE`, are one site.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters.

    Returns:
        Median nearest neighbor distance in meters, NaN with less than two sites.
    """
    distances = np.atleast_2d(distances)
    distances = np.where(distances > COLOCATED_DISTANCE, distances, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(np.nanmin(distances, axis=1)))


def station_density(distances: np.ndarray, radius: float = 0.0) -> np.ndarray:
    """Calculate the density of station sites around each station.

    A Gaussian kernel sums the neighboring sites; co-located sensors count as one
    site.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.
        radius: Standard deviation of the kernel in meters. If 0.0, the median
            nearest neighbor distance between sites.

    Returns:
        Array of shape (n_stations,) with the density at each station: 1.0 for a
            station without neighboring sites.
    """
    if radius <= 0.0:
        radius = nearest_neighbor_distance(distances)
    if not np.isfinite(radius):
        return np.ones(distances.shape[0])
    n_sensors = colocated_sensors(distances)
    sites = np.where(distances <= COLOCATED_DISTANCE, np.nan, distances)
    kernel = np.exp(-(sites**2) / (2 * radius**2)) / n_sensors[np.newaxis, :]
    return np.nansum(kernel, axis=1) + 1.0


def station_independence(distances: np.ndarray) -> np.ndarray:
    """Calculate how many independent stations each station counts as.

    An isolated station counts as one independent station. Stations in dense
    clusters count as a fraction of one: `1 - (k - k_min) / k_max` with the station
    density `k`. Co-located sensors share the count of their site.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.

    Returns:
        Array of shape (n_stations,) with values between 0 and 1.
    """
    density = station_density(distances)
    independence = 1.0 - (density - density.min()) / density.max()
    return independence / colocated_sensors(distances)


def independent_stations_distance(
    distances: np.ndarray,
    independence: np.ndarray,
    n_stations: float,
) -> np.ndarray:
    """Distance at which the closest stations add up to independent stations.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        independence: Array of shape (n_stations,) with the independent station
            count of each station.
        n_stations: Number of independent stations to reach.

    Returns:
        Array of shape (n_nodes, 1) with the distance of the station at which the
            closest stations reach `n_stations`, or of the most distant station if
            the network has fewer independent stations.
    """
    order = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, order, axis=1)
    cumulative = np.cumsum(independence[order], axis=1)

    idx = np.argmax(cumulative >= n_stations, axis=1)
    idx[cumulative[:, -1] < n_stations] = distances.shape[1] - 1
    return sorted_distances[np.arange(distances.shape[0]), idx, np.newaxis]


def weights_plateau_gaussian(
    distances: np.ndarray,
    independence: np.ndarray,
    plateau_stations: float = 3.0,
    taper_stations: float = 8.0,
) -> np.ndarray:
    """Gaussian taper beyond a plateau of the closest independent stations.

    The plateau ends where the closest stations add up to `plateau_stations`
    independent stations. The Gaussian taper starts there; its standard deviation
    is half the distance at which they add up to `taper_stations`.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        independence: Array of shape (n_stations,) with the independent station
            count of each station.
        plateau_stations: Independent stations of the plateau.
        taper_stations: Independent stations that set the taper width.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    plateau = independent_stations_distance(distances, independence, plateau_stations)
    taper = independent_stations_distance(distances, independence, taper_stations)
    sigma = taper / 2

    weights = np.exp(-((distances - plateau) ** 2) / (2 * sigma**2))
    weights[distances <= plateau] = 1.0
    return weights


def weights_log_logistic(
    distances: np.ndarray,
    plateau_distances: np.ndarray,
    taper_scale: float = 1.8,
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
    _interstation_distances: np.ndarray = PrivateAttr()
    _independence: dict[bytes, np.ndarray] = PrivateAttr(default_factory=dict)

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
        self._interstation_distances = interstation_distances(list(self._stations))
        self._independence = {}

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

    def get_interstation_distances(self, station_indices: np.ndarray) -> np.ndarray:
        """Get the distances between a set of stations.

        Args:
            station_indices: Indices of the stations.

        Returns:
            Array of shape (n_stations, n_stations) in meters, NaN on the diagonal.
        """
        return self._interstation_distances[station_indices][:, station_indices]

    def get_independence(self, station_indices: np.ndarray) -> np.ndarray:
        """Get the independent station count of a set of stations.

        The count depends on the available stations, it is cached for each set of
        stations.

        Args:
            station_indices: Indices of the available stations.

        Returns:
            Array of shape (n_stations,) with values between 0 and 1.
        """
        key = station_indices.tobytes()
        if key not in self._independence:
            self._independence[key] = station_independence(
                self.get_interstation_distances(station_indices)
            )
        return self._independence[key]

    def log_independent_stations(self) -> None:
        """Log how many independent stations the network counts as."""
        independence = self.get_independence(np.arange(len(self._stations)))
        logger.info(
            "the %d stations count as %.1f independent stations",
            independence.size,
            independence.sum(),
        )

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        """Calculate the weights from the node-station distances.

        Args:
            distances: Array of shape (n_nodes, n_stations) with node-station
                distances in meters.
            station_indices: Indices of the stations in the station list.
            nodes: The nodes of the distances.

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

        weights = self.calculate_weights(np.array(distances), station_indices, nodes)
        if self.waterlevel > 0.0:
            weights = (1 - self.waterlevel) * weights + self.waterlevel
        return weights


class DistanceWeights(StationWeights):
    """The closest stations get full weight, more distant stations a Gaussian taper.

    Close stations constrain the location of an event best. The taper width is
    absolute, the same for all nodes.
    """

    weights: Literal["DistanceWeights"] = "DistanceWeights"

    distance_taper: PositiveFloat | Literal["nearest_neighbor", "mean_interstation"] = (
        Field(
            default="nearest_neighbor",
            description=(
                "Full width at half maximum of the Gaussian taper in meters."
                f' `"nearest_neighbor"` uses {NEAREST_NEIGHBOR_TAPER:g} times the median'
                ' distance between neighboring station sites, `"mean_interstation"`'
                " twice the mean interstation distance of the network."
            ),
        )
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
        super().prepare(stations, octree)
        if self.distance_taper == "mean_interstation":
            self.distance_taper = 2 * stations.mean_interstation_distance()
        elif self.distance_taper == "nearest_neighbor":
            self.distance_taper = NEAREST_NEIGHBOR_TAPER * nearest_neighbor_distance(
                self._interstation_distances
            )
        self._distance_taper = self.distance_taper
        logger.info(
            "distance weighting uses %d closest stations and a taper of %g m",
            self.required_closest_stations,
            self._distance_taper,
        )

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        return weights_gaussian(
            distances,
            distance_taper=self._distance_taper,
            required_stations=self.required_closest_stations,
        )


class StationDensityWeights(StationWeights):
    """The closest independent stations get full weight, then a Gaussian taper.

    Every station counts as a number of independent stations: an isolated station
    as one, a station in a dense cluster as a fraction of one. The plateau of full
    weight and the taper are set in independent stations, so they adapt to the
    station spacing around each node.
    """

    weights: Literal["StationDensityWeights"] = "StationDensityWeights"

    plateau_stations: PositiveFloat = Field(
        default=3.0,
        description=PLATEAU_STATIONS_DESCRIPTION,
    )
    taper_stations: PositiveFloat = Field(
        default=8.0,
        description="Number of independent stations that sets the width of the"
        " Gaussian taper: its standard deviation is half the distance at which the"
        " closest stations add up to this number. If the network has fewer"
        " independent stations, the most distant station sets the width.",
    )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        super().prepare(stations, octree)
        self.log_independent_stations()

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        return weights_plateau_gaussian(
            distances,
            self.get_independence(station_indices),
            plateau_stations=self.plateau_stations,
            taper_stations=self.taper_stations,
        )


class LogLogisticWeights(StationWeights):
    """The closest independent stations set the plateau, then a log-logistic taper.

    The plateau distance of a node is the distance at which the closest stations add
    up to `plateau_stations` independent stations, as for `StationDensityWeights`.
    The weights decay with the distance in units of the plateau distance,
    `1 / (1 + (d / (taper_scale * d_plateau)) ** taper_exponent)`, like the
    confidence of the phase picks of small events.
    """

    weights: Literal["LogLogisticWeights"] = "LogLogisticWeights"

    plateau_stations: PositiveFloat = Field(
        default=4.0,
        description=PLATEAU_STATIONS_DESCRIPTION,
    )
    taper_scale: PositiveFloat = Field(
        default=1.8,
        description="Distance at which the weight is 0.5, in units of the plateau"
        " distance of the node.",
    )
    taper_exponent: PositiveFloat = Field(
        default=4.0,
        description="Exponent of the log-logistic taper. Higher values decay"
        " faster beyond `taper_scale` times the plateau distance.",
    )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        super().prepare(stations, octree)
        self.log_independent_stations()

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        plateau = independent_stations_distance(
            distances,
            self.get_independence(station_indices),
            self.plateau_stations,
        )
        return weights_log_logistic(
            distances,
            plateau,
            taper_scale=self.taper_scale,
            taper_exponent=self.taper_exponent,
        )


class LocationBalancedWeights(StationWeights):
    """Log-logistic weights for detection, declustered weights for the location.

    The root nodes of the octree detect events: they use the log-logistic weights,
    which favor the closest stations with the strongest phase confidences. The
    refined nodes locate the events: their weights are multiplied by the
    independent station count to the power of `location_declustering`, so that
    clusters of stations do not dominate the location, and their taper can be wider.
    """

    weights: Literal["LocationBalancedWeights"] = "LocationBalancedWeights"

    plateau_stations: PositiveFloat = Field(
        default=4.0,
        description=PLATEAU_STATIONS_DESCRIPTION,
    )
    taper_scale: PositiveFloat = Field(
        default=2.2,
        description="Distance at which the weight of the root nodes is 0.5, in units"
        " of the plateau distance of the node.",
    )
    taper_exponent: PositiveFloat = Field(
        default=4.0,
        description="Exponent of the log-logistic taper.",
    )
    location_taper_scale: PositiveFloat = Field(
        default=2.2,
        description="Distance at which the weight of the refined nodes is 0.5, in"
        " units of the plateau distance of the node.",
    )
    location_declustering: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Exponent of the independent station count in the weights of the"
        " refined nodes. With `0.0` all stations count fully, with `1.0` a station"
        " counts as its number of independent stations.",
    )

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        super().prepare(stations, octree)
        self.log_independent_stations()

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
        nodes: Sequence[Node],
    ) -> np.ndarray:
        independence = self.get_independence(station_indices)
        plateau = independent_stations_distance(
            distances, independence, self.plateau_stations
        )
        weights = weights_log_logistic(
            distances,
            plateau,
            taper_scale=self.taper_scale,
            taper_exponent=self.taper_exponent,
        )
        refined = np.array([node.level > 0 for node in nodes])
        if refined.any():
            weights[refined] = (
                weights_log_logistic(
                    distances[refined],
                    plateau[refined],
                    taper_scale=self.location_taper_scale,
                    taper_exponent=self.taper_exponent,
                )
                * independence[np.newaxis, :] ** self.location_declustering
            )
        return weights


# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type StationWeightsType = StationWeights
else:
    type StationWeightsType = Annotated[
        Union[StationWeights.get_subclasses()],
        Field(discriminator="weights"),
    ]
