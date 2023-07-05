from __future__ import annotations

import contextlib
import logging
import struct
from collections import defaultdict
from functools import cached_property
from hashlib import sha1
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import numpy as np
from pydantic import ConfigDict, BaseModel, Field, PositiveFloat, PrivateAttr
from pyrocko import orthodrome as od

from lassie.models.location import CoordSystem, Location

if TYPE_CHECKING:
    from typing_extensions import Self

    from lassie.models.station import Stations

logger = logging.getLogger(__name__)
KM = 1e3


def get_node_coordinates(
    nodes: Sequence[Node],
    system: CoordSystem = "geographic",
) -> np.ndarray:
    node_locations = (node.as_location() for node in nodes)
    if system == "geographic":
        return np.array(
            [
                (*node.effective_lat_lon, node.effective_elevation)
                for node in node_locations
            ]
        )
    if system == "cartesian":
        return np.array(
            [
                (node.east_shift, node.north_shift, node.effective_elevation)
                for node in node_locations
            ]
        )
    raise ValueError(f"Unknown coordinate system: {system}")


class NodeSplitError(Exception):
    ...


class Node(BaseModel):
    east: float
    north: float
    depth: float
    size: float
    semblance: float = 0.0

    tree: Octree | None = Field(None, exclude=True)
    children: tuple[Node] = Field((), exclude=True)

    _hash: bytes | None = PrivateAttr(None)
    _children_cached: tuple[Node] = PrivateAttr(())
    _location: Location | None = PrivateAttr(None)

    def split(self) -> tuple[Node]:
        if not self.tree:
            raise EnvironmentError("Parent tree is not set.")

        if not self.can_split():
            raise NodeSplitError("Cannot split node below limit.")

        if not self._children_cached:
            half_size = self.size / 2

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

    @property
    def distance_border(self) -> float:
        if not self.tree:
            raise AttributeError("parent tree not set")
        tree = self.tree
        return min(
            self.north - tree.north_bounds[0],
            tree.north_bounds[1] - self.north,
            self.east - tree.east_bounds[0],
            tree.east_bounds[1] - self.east,
            self.depth - tree.depth_bounds[0],
            tree.depth_bounds[1] - self.depth,
        )

    def can_split(self) -> bool:
        half_size = self.size / 2
        return half_size >= self.tree.size_limit

    def reset(self) -> None:
        self._children_cached = self.children
        self.children = ()
        self.semblance = 0.0

    def set_parent(self, tree: Octree) -> None:
        self.tree = tree
        for child in self.children:
            child.set_parent(tree)

    def distance_to_location(self, location: Location) -> float:
        return location.distance_to(self.as_location())

    def as_location(self) -> Location:
        if not self._location:
            self._location = Location.construct(
                lat=self.tree.center_lat,
                lon=self.tree.center_lon,
                elevation=self.tree.surface_elevation,
                east_shift=float(self.east),
                north_shift=float(self.north),
                depth=float(self.depth),
            )
        return self._location

    def __iter__(self) -> Iterator[Node]:
        if self.children:
            for child in self.children:
                yield from child
        else:
            yield self

    def hash(self) -> bytes:
        if self._hash is None:
            self._hash = sha1(
                struct.pack(
                    "dddddd",
                    self.tree.center_lat,
                    self.tree.center_lon,
                    self.east,
                    self.north,
                    self.depth,
                    self.size,
                )
            ).digest()
        return self._hash

    def __hash__(self) -> int:
        return hash(self.hash())


class Octree(BaseModel):
    center_lat: float = 0.0
    center_lon: float = 0.0
    surface_elevation: float = 0.0
    size_initial: PositiveFloat = 2 * KM
    size_limit: PositiveFloat = 500
    east_bounds: tuple[float, float] = (-10 * KM, 10 * KM)
    north_bounds: tuple[float, float] = (-10 * KM, 10 * KM)
    depth_bounds: tuple[float, float] = (0 * KM, 20 * KM)
    absorbing_boundary: float = 1 * KM

    nodes: list[Node] | None = None

    _root_nodes: list[Node] = PrivateAttr([])
    _cached_coordinates: dict[CoordSystem, np.ndarray] = PrivateAttr({})
    model_config = ConfigDict(ignored_types=(cached_property,))

    def __init__(self, **data) -> None:
        super().__init__(**data)

        if self.nodes:
            logger.debug("loading existing nodes")
            self._root_nodes = self.nodes
            for node in self._root_nodes:
                node.tree = self
        else:
            logger.debug("initializing root nodes")
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

    @cached_property
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
        with contextlib.suppress(AttributeError):
            del self.n_nodes

    def reset(self) -> None:
        logger.debug("resetting tree")
        self._clear_cache()
        self._root_nodes = self._get_root_nodes(self.size_initial)

    def reduce_surface(self, accumulator: Callable = np.max) -> np.ndarray:
        groups = defaultdict(list)
        for node in self:
            groups[(node.east, node.north, node.size)].append(node.semblance)

        values = (accumulator(values) for values in groups.values())
        return np.array(
            [(*node, val) for node, val in zip(groups.keys(), values, strict=True)]
        )

    @property
    def semblance(self) -> np.ndarray:
        return np.fromiter((node.semblance for node in self), float)

    def map_semblance(self, semblance: np.ndarray) -> None:
        for node, node_semblance in zip(self, semblance, strict=True):
            node.semblance = float(node_semblance)

    def get_coordinates(self, system: CoordSystem = "geographic") -> np.ndarray:
        if self._cached_coordinates.get(system) is None:
            nodes = (node for node in self)
            self._cached_coordinates[system] = get_node_coordinates(
                nodes, system=system
            )
        return self._cached_coordinates[system]

    def distances_stations(self, stations: Stations) -> np.ndarray:
        """Returns the distances from all nodes to all stations.

        Args:
            stations (Stations): Stations to calculate distance to.

        Returns:
            np.ndarray: Of shape (n-nodes, n-stations).
        """
        node_coords = self.get_coordinates(system="geographic")
        sta_coords = stations.get_coordinates(system="geographic")

        sta_coords = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T
        return np.linalg.norm(sta_coords - node_coords[:, np.newaxis], axis=2)

    def get_nodes(self, semblance_threshold: float = 0.0) -> list[Node]:
        """Get all nodes with a semblance above a threshold.

        Args:
            semblance_threshold (float): Semblance threshold. Default is 0.0.

        Returns:
            list[Node]: List of nodes.
        """
        if not semblance_threshold:
            return list(self)
        return [node for node in self if node.semblance >= semblance_threshold]

    def is_node_in_bounds(self, node: Node) -> bool:
        """Check if node is inside the absorbing boundary.

        Args:
            node (Node): Node to check.

        Returns:
            bool: Check if node is absorbed.
        """
        return node.distance_border > self.absorbing_boundary

    def make_concrete(self) -> None:
        """Make octree concrete for serialisation."""
        self._root_nodes = self.get_nodes()
        self.nodes = self._root_nodes

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

    def get_smallest_node_size(self) -> float:
        return min(node.size for node in self)

    def copy(self, deep=False) -> Self:
        tree = super().copy(deep=deep)
        tree._clear_cache()
        for node in tree._root_nodes:
            node.set_parent(tree)
        return tree

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
