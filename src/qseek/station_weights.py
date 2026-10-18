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
# Rows of the interstation distances calculated at once
INTERSTATION_BLOCK = 256
# Taper width of DistanceWeights in units of the nearest neighbor distance; the
# description of DistanceWeights.distance_taper repeats the value
NEAREST_NEIGHBOR_TAPER = 6.0


def weights_gaussian(
    distances: np.ndarray,
    distance_taper: float,
    required_stations: int = 4,
) -> np.ndarray:
    """Gaussian distance weights with a plateau of the closest stations.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        distance_taper: Full width at half maximum of the Gaussian taper in meters.
        required_stations: Number of closest stations with full weight.

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
    return weights


def interstation_distances(locations: Sequence[Location]) -> np.ndarray:
    """Calculate the distances between stations.

    Args:
        locations: Station locations.

    Returns:
        Array of shape (n_stations, n_stations) with the interstation distances in
            meters, float32. The diagonal is NaN.
    """
    coords = get_coordinates(locations, system="geographic")
    coords_ecef = np.array(od.geodetic_to_ecef(*coords.T)).T
    n_stations = coords_ecef.shape[0]
    distances = np.empty((n_stations, n_stations), dtype=np.float32)
    # Row blocks keep the temporary differences small for large networks
    for start in range(0, n_stations, INTERSTATION_BLOCK):
        block = coords_ecef[start : start + INTERSTATION_BLOCK]
        distances[start : start + block.shape[0]] = np.linalg.norm(
            block[:, np.newaxis] - coords_ecef[np.newaxis], axis=2
        )
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nearest = np.nanmin(
            distances, axis=1, where=distances > COLOCATED_DISTANCE, initial=np.inf
        )
        nearest[~np.isfinite(nearest)] = np.nan
        return float(np.nanmedian(nearest))


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
    n_sensors = colocated_sensors(distances).astype(distances.dtype)
    density = np.empty(distances.shape[0], dtype=distances.dtype)
    # Row blocks keep the temporary kernel small for large networks
    for start in range(0, distances.shape[0], INTERSTATION_BLOCK):
        kernel = np.array(distances[start : start + INTERSTATION_BLOCK], copy=True)
        kernel[kernel <= COLOCATED_DISTANCE] = np.nan
        np.square(kernel, out=kernel)
        kernel *= -1.0 / (2 * radius**2)
        np.exp(kernel, out=kernel)
        kernel /= n_sensors[np.newaxis, :]
        density[start : start + kernel.shape[0]] = np.nansum(kernel, axis=1)
    return density + 1.0


def independent_station_counts(distances: np.ndarray) -> np.ndarray:
    """Calculate how many independent stations each station counts as.

    The count is `1 - (k - k_min) / k_max` with the station density `k` and its
    lowest and highest value in the network, `k_min` and `k_max`. The station with
    the lowest density counts as one independent station, an isolated station
    (`k = 1`) as well. Dividing by `k_max` instead of `k_max - k_min` keeps the
    densest station at `k_min / k_max` instead of zero, so that every station
    counts. The counts are relative to the network: adding or removing a station
    changes the counts of all stations. Co-located sensors share the count of their
    site.

    Args:
        distances: Array of shape (n_stations, n_stations) with interstation
            distances in meters, NaN on the diagonal.

    Returns:
        Array of shape (n_stations,) with values between 0 and 1.
    """
    density = station_density(distances)
    counts = 1.0 - (density - density.min()) / density.max()
    return counts / colocated_sensors(distances)


def weights_gaussian_density(
    distances: np.ndarray,
    factors: np.ndarray,
    distance_taper: float,
    required_stations: int = 4,
) -> np.ndarray:
    """Gaussian distance weights that approach a station factor in the taper.

    The weight is `g * (f + (1 - f) * g)` with the Gaussian distance weight `g` and
    the station factor `f`: 1 on the plateau of the closest stations, `g * f` far
    away. The weights are continuous where `g` is.

    Args:
        distances: Array of shape (n_nodes, n_stations) with node-station distances
            in meters.
        factors: Array of shape (n_stations,) with the factor of each station
            between 0 and 1, e.g. the independent station count.
        distance_taper: Full width at half maximum of the Gaussian taper in meters.
        required_stations: Number of closest stations with full weight.

    Returns:
        Array of shape (n_nodes, n_stations) with weights between 0 and 1.
    """
    weights = weights_gaussian(distances, distance_taper, required_stations)
    factors = factors.astype(weights.dtype)[np.newaxis, :]
    return weights * (factors + (1.0 - factors) * weights)


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
    _station_counts: dict[bytes, np.ndarray] = PrivateAttr(default_factory=dict)

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
        self._station_counts = {}

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
        if self._interstation_distances is None:
            self._interstation_distances = interstation_distances(list(self._stations))
        n_stations = self._interstation_distances.shape[0]
        if station_indices.size == n_stations and np.array_equal(
            station_indices, np.arange(n_stations)
        ):
            return self._interstation_distances
        return self._interstation_distances[np.ix_(station_indices, station_indices)]

    def get_station_counts(self, station_indices: np.ndarray) -> np.ndarray:
        """Get the independent station counts of a set of stations.

        The counts depend on the available stations, they are cached for each set
        of stations.

        Args:
            station_indices: Indices of the available stations.

        Returns:
            Array of shape (n_stations,) with values between 0 and 1.
        """
        key = station_indices.tobytes()
        if key not in self._station_counts:
            self._station_counts[key] = independent_station_counts(
                self.get_interstation_distances(station_indices)
            )
        return self._station_counts[key]

    def resolve_distance_taper(
        self,
        distance_taper: float | Literal["mean_interstation", "nearest_neighbor"],
        stations: StationInventory,
    ) -> float:
        """Resolve the width of a Gaussian distance taper for the stations.

        Args:
            distance_taper: Full width at half maximum in meters,
                `"mean_interstation"` for twice the mean interstation distance or
                `"nearest_neighbor"` for `NEAREST_NEIGHBOR_TAPER` times the median
                distance between neighboring sites.
            stations: Stations of the search.

        Returns:
            Full width at half maximum in meters, infinite with fewer than two
                station sites.
        """
        if distance_taper == "mean_interstation":
            taper = (
                2 * stations.mean_interstation_distance()
                if stations.n_stations > 1
                else np.inf
            )
        elif distance_taper == "nearest_neighbor":
            all_stations = np.arange(len(self._stations))
            taper = NEAREST_NEIGHBOR_TAPER * nearest_neighbor_distance(
                self.get_interstation_distances(all_stations)
            )
        else:
            taper = float(distance_taper)
        if not np.isfinite(taper) or taper <= 0.0:
            # Fewer than two sites: all stations get full weight
            taper = np.inf
        return taper

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
    absolute, the same for all nodes. With the defaults, these are the station
    weights of earlier Qseek versions.
    """

    weights: Literal["DistanceWeights"] = "DistanceWeights"

    distance_taper: PositiveFloat | Literal["mean_interstation", "nearest_neighbor"] = (
        Field(
            default="mean_interstation",
            description=(
                "Full width at half maximum of the Gaussian taper in meters."
                ' `"mean_interstation"` uses twice the mean interstation distance of'
                ' the network, `"nearest_neighbor"` 6 times the median distance'
                " between neighboring station sites. Do not use"
                ' `"nearest_neighbor"` for networks with a dense array: its station'
                " spacing sets the median."
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
        self._distance_taper = self.resolve_distance_taper(
            self.distance_taper, stations
        )
        logger.info(
            "distance weights use %d closest stations and a taper of %g m",
            self.required_closest_stations,
            self._distance_taper,
        )

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


class DensityTaperWeights(StationWeights):
    """Distance weights that count distant clusters of stations as fewer stations.

    The closest stations get full weight and more distant stations the Gaussian
    taper of `DistanceWeights`. Seen from a distance, the stations of a dense
    cluster record the event in nearly the same direction: in the taper, each
    station's weight approaches its independent station count. The closest
    stations keep their full weight, also within a cluster.
    """

    weights: Literal["DensityTaperWeights"] = "DensityTaperWeights"

    distance_taper: PositiveFloat | Literal["mean_interstation", "nearest_neighbor"] = (
        Field(
            default="mean_interstation",
            description=(
                "Full width at half maximum of the Gaussian taper in meters."
                ' `"mean_interstation"` uses twice the mean interstation distance of'
                ' the network, `"nearest_neighbor"` 6 times the median distance'
                " between neighboring station sites."
            ),
        )
    )
    required_closest_stations: PositiveInt = Field(
        default=4,
        description="Number of closest stations of a node that get full weight.",
    )
    station_density: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Exponent of the independent station count in the taper. A"
        " station in a dense cluster counts as a fraction of an independent"
        " station; with `1.0` a distant cluster weighs as much as its independent"
        " stations, with `0.0` these are the `DistanceWeights`.",
    )

    _distance_taper: float = PrivateAttr()

    def prepare(self, stations: StationInventory, octree: Octree) -> None:
        super().prepare(stations, octree)
        self._distance_taper = self.resolve_distance_taper(
            self.distance_taper, stations
        )
        counts = self.get_station_counts(np.arange(len(self._stations)))
        logger.info(
            "the %d stations count as %.1f independent stations; %d closest stations"
            " get full weight, then a taper of %g m",
            counts.size,
            counts.sum(),
            self.required_closest_stations,
            self._distance_taper,
        )

    def calculate_weights(
        self,
        distances: np.ndarray,
        station_indices: np.ndarray,
    ) -> np.ndarray:
        return weights_gaussian_density(
            distances,
            self.get_station_counts(station_indices) ** self.station_density,
            distance_taper=self._distance_taper,
            required_stations=self.required_closest_stations,
        )


# Statically the base class, pydantic validates the registered subclasses
if TYPE_CHECKING:
    type StationWeightsType = StationWeights
else:
    type StationWeightsType = Annotated[
        Union[StationWeights.get_subclasses()],
        Field(discriminator="weights"),
    ]
