from __future__ import annotations

import contextlib
import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, reduce
from hashlib import sha1
from operator import mul
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

from qseek.models.location import CoordSystem, Location

if TYPE_CHECKING:
    from typing_extensions import Self

    from qseek.models.station import Stations

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


@dataclass(slots=True)
class Node:
    east: float
    north: float
    depth: float
    size: float
    semblance: float = 0.0

    tree: Octree | None = None
    children: tuple[Node, ...] = ()

    _hash: bytes | None = None
    _children_cached: tuple[Node, ...] = ()
    _location: Location | None = None

    def split(self) -> tuple[Node, ...]:
        """Split the node into 8 children"""
        if not self.tree:
            raise EnvironmentError("Parent tree is not set.")

        if not self.can_split():
            raise NodeSplitError("Cannot split node below limit.")

        if not self._children_cached:
            half_size = self.size / 2

            self._children_cached = tuple(
                Node(
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
        """Distance to the closest EW, NS or bottom border of the tree.

        !!! note
            Surface distance is excluded.

        Raises:
            AttributeError: If the parent tree is not set.

        Returns:
            float: Distance to the closest border.
        """
        if not self.tree:
            raise AttributeError("parent tree not set")
        tree = self.tree
        return min(
            self.north - tree.north_bounds[0],
            tree.north_bounds[1] - self.north,
            self.east - tree.east_bounds[0],
            tree.east_bounds[1] - self.east,
            tree.depth_bounds[1] - self.depth,
        )

    def can_split(self) -> bool:
        """Check if the node can be split.

        Raises:
            AttributeError: If the parent tree is not set.

        Returns:
            bool: True if the node can be split.
        """
        if self.tree is None:
            raise AttributeError("parent tree not set")
        return self.size > self.tree.smallest_node_size()

    def reset(self) -> None:
        """Reset the node to its initial state."""
        self._children_cached = self.children
        self.children = ()
        self.semblance = 0.0

    def set_parent(self, tree: Octree) -> None:
        """Set the parent tree of the node.

        Args:
            tree (Octree): The parent tree.
        """
        self.tree = tree
        for child in self.children:
            child.set_parent(tree)

    def distance_to_location(self, location: Location) -> float:
        """Distance to a location.

        Args:
            location (Location): Location to calculate distance to.

        Returns:
            float: Distance to location.
        """
        return location.distance_to(self.as_location())

    def as_location(self) -> Location:
        """Returns the location of the node.

        Returns:
            Location: Location of the node.
        """
        if not self.tree:
            raise AttributeError("parent tree not set")
        if not self._location:
            reference = self.tree.location
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
        if not self.tree:
            raise AttributeError("parent tree not set")
        if self._hash is None:
            self._hash = sha1(
                struct.pack(
                    "dddddd",
                    self.tree.location.lat,
                    self.tree.location.lon,
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
    location: Location = Field(
        default=Location(lat=0.0, lon=0.0),
        description="The reference location of the octree.",
    )
    size_initial: PositiveFloat = Field(
        default=2 * KM,
        description="Initial size of a cubic octree node in meters.",
    )
    size_limit: PositiveFloat = Field(
        default=500.0,
        description="Smallest possible size of an octree node in meters.",
    )
    east_bounds: tuple[float, float] = Field(
        default=(-10 * KM, 10 * KM),
        description="East bounds of the octree in meters.",
    )
    north_bounds: tuple[float, float] = Field(
        default=(-10 * KM, 10 * KM),
        description="North bounds of the octree in meters.",
    )
    depth_bounds: tuple[float, float] = Field(
        default=(0 * KM, 20 * KM),
        description="Depth bounds of the octree in meters.",
    )
    absorbing_boundary: float = Field(
        default=1 * KM,
        ge=0.0,
        description="Absorbing boundary in meters. Detections inside the boundary will be tagged.",
    )

    _root_nodes: list[Node] = PrivateAttr([])
    _cached_coordinates: dict[CoordSystem, np.ndarray] = PrivateAttr({})

    model_config = ConfigDict(ignored_types=(cached_property,))

    @field_validator("location")
    def check_reference(cls, location: Location) -> Location:  # noqa: N805
        if location.lat == 0.0 and location.lon == 0.0:
            raise ValueError("invalid  location, expected non-zero lat/lon")
        return location

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
                f"invalid octree size limits ({self.size_initial}, {self.size_limit}):"
                " Expected size_limit <= size_initial"
            )
        for dimension, ext in zip(
            ("east", "north", "depth"), self.extent(), strict=True
        ):
            if ext % self.size_initial:
                raise ValueError(
                    f"invalid octree initial_size {self.size_initial}:"
                    f" Not a multiple in {dimension} with size {ext}"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize octree. This method is called by the pydantic model"""
        self._root_nodes = self._get_root_nodes(self.size_initial)
        logger.info(
            "initializing octree volume with %d nodes and %.1f km³,"
            " smallest node size: %.1f m",
            self.n_nodes,
            self.volume / (KM**3),
            self.smallest_node_size(),
        )

    def extent(self) -> tuple[float, float, float]:
        """Returns the extent of the octree.

        Returns:
            tuple[float, float, float]: EW, NS and depth extent of the octree in meters.
        """
        return (
            self.east_bounds[1] - self.east_bounds[0],
            self.north_bounds[1] - self.north_bounds[0],
            self.depth_bounds[1] - self.depth_bounds[0],
        )

    def _get_root_nodes(self, length: float) -> list[Node]:
        ln = length
        ext_east, ext_north, ext_depth = self.extent()
        # FIXME: this is not correct, the nodes should be centered
        east_nodes = np.arange(ext_east // ln) * ln + ln / 2 + self.east_bounds[0]
        north_nodes = np.arange(ext_north // ln) * ln + ln / 2 + self.north_bounds[0]
        depth_nodes = np.arange(ext_depth // ln) * ln + ln / 2 + self.depth_bounds[0]

        return [
            Node(east=east, north=north, depth=depth, size=ln, tree=self)
            for east in east_nodes
            for north in north_nodes
            for depth in depth_nodes
        ]

    @cached_property
    def n_nodes(self) -> int:
        """Number of nodes in the octree"""
        return sum(1 for _ in self)

    @property
    def volume(self) -> float:
        """Volume of the octree in cubic meters"""
        return reduce(mul, self.extent())

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

    def reset(self) -> Self:
        """Reset the octree to its initial state"""
        logger.debug("resetting tree")
        self._clear_cache()
        self._root_nodes = self._get_root_nodes(self.size_initial)
        return self

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
        return self.size_initial / (2 ** self.n_levels())

    def n_levels(self) -> int:
        """Returns the number of levels in the octree.

        Returns:
            int: Number of levels.
        """
        return int(np.floor(np.log2(self.size_initial / self.size_limit)))

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
