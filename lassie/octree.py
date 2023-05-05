from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from pydantic import BaseModel, PositiveFloat, PrivateAttr
from pyrocko import orthodrome as od

from lassie.models.location import CoordSystem, Location

if TYPE_CHECKING:
    from lassie.models.station import Station, Stations

logger = logging.getLogger(__name__)
km = 1e3


class Node(BaseModel):
    east: float
    north: float
    depth: float
    size: float
    tree: Octree
    semblance: float = 0.0

    children: tuple[Node] = tuple()

    _children_cached: tuple[Node] = PrivateAttr(tuple())
    _location: Location | None = PrivateAttr(None)

    def split(self) -> tuple[Node]:
        if not self.tree:
            raise EnvironmentError("Parent tree is not set.")

        if not self._children_cached:
            half_size = self.size / 2
            if half_size < self.tree.size_limit:
                raise ValueError("Cannot split node below limit.")

            self._children_cached = tuple(
                Node.construct(
                    east=self.east + east * half_size / 2,
                    north=self.north + north * half_size / 2,
                    depth=self.depth + depth * half_size / 2,
                    size=half_size,
                    tree=self.tree,
                )
                for east in (-1, 1)
                for north in (-1, 1)
                for depth in (-1, 1)
            )

        self.children = self._children_cached
        if self.tree:
            self.tree._clear_cache()
        return self.children

    @property
    def coordinates(self) -> tuple[float, float, float]:
        return self.east, self.north, self.depth

    def reset(self) -> None:
        self._children_cached = self.children
        self.children = tuple()
        self.semblance = 0.0

    def distance_station(self, station: Station) -> float:
        return station.distance_to(self.as_location())

    def as_location(self) -> Location:
        if not self._location:
            self._location = Location.construct(
                lat=self.tree.center_lat,
                lon=self.tree.center_lon,
                elevation=self.tree.surface_elevation,
                east_shift=self.east,
                north_shift=self.north,
                depth=self.depth,
            )
        return self._location

    def __iter__(self) -> Iterator[Node]:
        if self.children:
            for child in self.children:
                yield from child
        else:
            yield self

    def __hash__(self) -> int:
        return hash(
            (
                self.tree.center_lat,
                self.tree.center_lon,
                self.east,
                self.north,
                self.depth,
                self.size,
            )
        )


class Octree(BaseModel):
    center_lat: float = 0.0
    center_lon: float = 0.0
    surface_elevation: float = 0.0
    size_initial: PositiveFloat = 2 * km
    size_limit: PositiveFloat = 500
    east_bounds: tuple[float, float] = (-10 * km, 10 * km)
    north_bounds: tuple[float, float] = (-10 * km, 10 * km)
    depth_bounds: tuple[float, float] = (0 * km, 20 * km)
    nodes: list[Node] = []

    _root_nodes: list[Node] = PrivateAttr([])
    _cached_coordinates: dict[CoordSystem, np.ndarray] = PrivateAttr({})

    def __init__(self, **data) -> None:
        super().__init__(**data)

        logger.debug("initializing nodes")
        self._root_nodes = self._get_root_nodes(self.size_initial)

    # @validator("east_bounds", "north_bounds", "depth_bounds", )
    # def _check_bounds(cls, bounds):  # noqa: N805
    #     for value in bounds:
    #         if value[0] >= value[1]:
    #             raise ValueError(f"invalid bounds {value}")
    #     return bounds

    # @validator("size_initial", "size_limit")
    # def _check_limits(cls, limits):  # noqa: N805
    #     if limits[0] < limits[1]:
    #         raise ValueError(f"invalid octree size limits {limits}")
    #     return limits

    def extent(self) -> tuple[float, float, float]:
        return (
            self.east_bounds[1] - self.east_bounds[0],
            self.north_bounds[1] - self.north_bounds[0],
            self.depth_bounds[1] - self.depth_bounds[0],
        )

    def _get_root_nodes(self, size: float) -> list[Node]:
        len = size
        ext_east, ext_north, ext_depth = self.extent()
        east_nodes = np.arange(ext_east // len) * len + len / 2 + self.east_bounds[0]
        north_nodes = np.arange(ext_north // len) * len + len / 2 + self.north_bounds[0]
        depth_nodes = np.arange(ext_depth // len) * len + len / 2 + self.depth_bounds[0]

        return [
            Node.construct(east=east, north=north, depth=depth, size=len, tree=self)
            for east in east_nodes
            for north in north_nodes
            for depth in depth_nodes
        ]

    @property
    def n_nodes(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[Node]:
        for node in self._root_nodes:
            yield from node

    def __getitem__(self, idx: int) -> Node:
        for inode, node in enumerate(self):
            if inode == idx:
                return node
        raise IndexError(f"bad node index {idx}")

    def _clear_cache(self) -> None:
        self._cached_coordinates.clear()

    def reset(self) -> None:
        logger.debug("resetting tree")
        self._clear_cache()
        self._root_nodes = self._get_root_nodes(self.size_initial)

    def reduce_surface(self, accumulator: Callable = np.max) -> np.ndarray:
        groups = defaultdict(list)
        for node in self:
            groups[(node.east, node.north)].append(node.semblance)

        values = (accumulator(values) for values in groups.values())
        return np.array(
            [(east, north, val) for (east, north), val in zip(groups.keys(), values)]
        )

    @property
    def semblance(self) -> np.ndarray:
        return np.fromiter((node.semblance for node in self), float)

    def add_semblance(self, semblance: np.ndarray) -> None:
        n_nodes = 0
        for node, node_semblance in zip(self, semblance):
            node.semblance = float(node_semblance)
            n_nodes += 1
        if n_nodes != semblance.size:
            raise ValueError(
                f"semblance is of bad shape {semblance.shape}, expected {n_nodes}"
            )

    def get_coordinates(self, system: CoordSystem = "geographic") -> np.ndarray:
        if self._cached_coordinates.get(system) is None:
            node_locations = [node.as_location() for node in self]

            if system == "geographic":
                self._cached_coordinates[system] = np.array(
                    [
                        (*node.effective_lat_lon, node.effective_elevation)
                        for node in node_locations
                    ]
                )
            elif system == "cartesian":
                self._cached_coordinates[system] = np.array(
                    [
                        (node.east_shift, node.north_shift, node.effective_elevation)
                        for node in node_locations
                    ]
                )
        return self._cached_coordinates[system]

    def distances_stations(self, stations: Stations) -> np.ndarray:
        """Returns the distances from all nodes to all stations.

        Args:
            stations (Stations): Stations to calculate distance to.

        Returns:
            np.ndarray: Of shape n-nodes, n-stations.
        """
        node_coords = self.get_coordinates(system="geographic")
        sta_coords = stations.get_coordinates(system="geographic")

        sta_coords = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T
        return np.linalg.norm(sta_coords - node_coords[:, np.newaxis], axis=2)

    def refine(self, semblance_threshold: float) -> list[Node]:
        new_nodes = []
        for node in self:
            if node.semblance > semblance_threshold:
                new_nodes.extend(node.split())
        return new_nodes

    def finest_tree(self) -> Octree:
        tree = self.copy()
        tree.reset()
        tree._root_nodes = tree._get_root_nodes(tree.size_limit)
        return tree

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = self.get_coordinates("cartesian").T
        ax.scatter(coords[0], coords[1], coords[2], c=self.semblance, cmap="Oranges")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.set_zlabel("depth [m]")
        plt.show()

    def plot_surface(self) -> None:
        import matplotlib.pyplot as plt

        surface = self.reduce_surface()
        ax = plt.figure().gca()
        ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 2], cmap="Oranges")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        plt.show()

    def __hash__(self) -> int:
        return hash(
            (
                self.size_initial,
                self.size_limit,
                self.east_bounds,
                self.north_bounds,
                self.depth_bounds,
            )
        )
