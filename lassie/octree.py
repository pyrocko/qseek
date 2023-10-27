from __future__ import annotations

import contextlib
import logging
import struct
from collections import defaultdict
from functools import cached_property
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    field_validator,
    model_validator,
)
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
    children: tuple[Node, ...] = Field(default=(), exclude=True)

    _hash: bytes | None = PrivateAttr(None)
    _children_cached: tuple[Node, ...] = PrivateAttr(())
    _location: Location | None = PrivateAttr(None)

    def split(self) -> tuple[Node, ...]:
        if not self.tree:
            raise EnvironmentError("Parent tree is not set.")

        if not self.can_split():
            raise NodeSplitError("Cannot split node below limit.")

        if not self._children_cached:
            half_size = self.size / 2

            self._children_cached = tuple(
                Node.model_construct(
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
        if self.tree is None:
            raise AttributeError("parent tree not set")
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
        if not self.tree:
            raise AttributeError("parent tree not set")
        if not self._location:
            reference = self.tree.reference
            self._location = Location.model_construct(
                lat=reference.lat,
                lon=reference.lon,
                elevation=reference.elevation,
                east_shift=reference.east_shift + float(self.east),
                north_shift=reference.north_shift + float(self.north),
                depth=reference.depth + float(self.depth),
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
                    self.tree.reference.lat,
                    self.tree.reference.lon,
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
    reference: Location = Location(lat=0.0, lon=0)
    size_initial: PositiveFloat = 2 * KM
    size_limit: PositiveFloat = 500
    east_bounds: tuple[float, float] = (-10 * KM, 10 * KM)
    north_bounds: tuple[float, float] = (-10 * KM, 10 * KM)
    depth_bounds: tuple[float, float] = (0 * KM, 20 * KM)
    absorbing_boundary: float = Field(default=1 * KM, ge=0.0)

    _root_nodes: list[Node] = PrivateAttr([])
    _cached_coordinates: dict[CoordSystem, np.ndarray] = PrivateAttr({})

    model_config = ConfigDict(ignored_types=(cached_property,))

    @field_validator("east_bounds", "north_bounds", "depth_bounds")
    def check_bounds(
        cls,  # noqa: N805
        bounds: tuple[float, float],
    ) -> tuple[float, float]:
        if bounds[0] >= bounds[1]:
            raise ValueError(f"invalid bounds {bounds}, expected (min, max)")
        return bounds

    @model_validator(mode="after")
    def check_limits(self) -> Octree:
        """Check that the size limits are valid."""
        if self.size_limit > self.size_initial:
            raise ValueError(
                f"invalid octree size limits ({self.size_initial}, {self.size_limit}),"
                " expected size_limit <= size_initial"
            )
        self.reference = self.reference.shifted_origin()
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize octree. This method is called by the pydantic model"""
        logger.info(
            "initializing octree, smallest node size: %.1f m",
            self.smallest_node_size(),
        )
        self._root_nodes = self._get_root_nodes(self.size_initial)

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
            Node.model_construct(
                east=east, north=north, depth=depth, size=len, tree=self
            )
            for east in east_nodes
            for north in north_nodes
            for depth in depth_nodes
        ]

    @cached_property
    def n_nodes(self) -> int:
        """Number of nodes in the octree"""
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
        """Reset the octree to its initial state"""
        logger.debug("resetting tree")
        self._clear_cache()
        self._root_nodes = self._get_root_nodes(self.size_initial)

    def reduce_surface(self, accumulator: Callable = np.max) -> np.ndarray:
        """Reduce the octree's nodes to the surface

        Args:
            accumulator (Callable, optional): Accumulator function. Defaults to np.max.

        Returns:
            np.ndarray: Of shape (n-nodes, 4) with columns (east, north, depth, value).
        """
        groups = defaultdict(list)
        for node in self:
            groups[(node.east, node.north, node.size)].append(node.semblance)

        values = (accumulator(values) for values in groups.values())
        return np.array(
            [(*node, val) for node, val in zip(groups.keys(), values, strict=True)]
        )

    @property
    def semblance(self) -> np.ndarray:
        """Returns the semblance values of all nodes."""
        return np.array([node.semblance for node in self])

    def map_semblance(self, semblance: np.ndarray) -> None:
        """Maps semblance values to nodes.

        Args:
            semblance (np.ndarray): Of shape (n-nodes,).
        """
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

    def smallest_node_size(self) -> float:
        """Returns the smallest possible node size.

        Returns:
            float: Smallest possible node size.
        """
        size = self.size_initial
        while size >= self.size_limit * 2:
            size /= 2
        return size

    def n_levels(self) -> int:
        """Returns the number of levels in the octree.

        Returns:
            int: Number of levels.
        """
        levels = 0
        size = self.size_initial
        while size >= self.size_limit * 2:
            levels += 1
            size /= 2
        return levels

    def total_number_nodes(self) -> int:
        """Returns the total number of nodes of all levels.

        Returns:
            int: Total number of nodes.
        """
        return len(self._root_nodes) * (8 ** self.n_levels())

    def cached_bottom(self) -> Self:
        """Returns a copy of the octree refined to the cached bottom nodes.

        Raises:
            EnvironmentError: If the octree has never been split.

        Returns:
            Self: Copy of the octree with cached bottom nodes.
        """
        tree = self.copy(deep=True)
        split_nodes = []
        for node in tree:
            if node._children_cached:
                split_nodes.extend(node.split())
        if not split_nodes:
            raise EnvironmentError("octree has never been split.")
        return tree

    def copy(self, deep=False) -> Self:
        tree = super().model_copy(deep=deep)
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
