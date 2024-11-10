from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, reduce
from hashlib import sha1
from operator import mul
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Sequence

import numpy as np
import scipy.interpolate
import scipy.optimize
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
from qseek.utils import Range

if TYPE_CHECKING:
    from typing_extensions import Self

    from qseek.models.station import Stations

logger = logging.getLogger(__name__)

KM = 1e3


def get_node_coordinates(
    nodes: Iterable[Node],
    system: CoordSystem = "geographic",
) -> np.ndarray:
    if system == "geographic":
        node_locations = (node.as_location() for node in nodes)
        return np.array(
            [
                (*node.effective_lat_lon, node.effective_elevation)
                for node in node_locations
            ]
        )
    if system == "cartesian":
        node_locations = (node.as_location() for node in nodes)
        return np.array(
            [
                (node.east_shift, node.north_shift, node.effective_elevation)
                for node in node_locations
            ]
        )
    if system == "raw":
        return np.array([(node.east, node.north, node.depth) for node in nodes])
    raise ValueError(f"Unknown coordinate system: {system}")


def distances_stations(nodes: Sequence[Node], stations: Stations) -> np.ndarray:
    """Returns the 3D distances from all nodes to all stations.

    Args:
        nodes (Sequence[Node]): Nodes to calculate distance from.
        stations (Stations): Stations to calculate distance to.

    Returns:
        np.ndarray: Of shape (n-nodes, n-stations).
    """
    node_coords = get_node_coordinates(nodes, system="geographic")
    sta_coords = stations.get_coordinates(system="geographic")

    sta_coords = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
    node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T
    return np.linalg.norm(sta_coords - node_coords[:, np.newaxis], axis=2)


class NodeSplitError(Exception): ...


@dataclass(slots=True)
class Node:
    east: float
    north: float
    depth: float
    size: float
    level: int
    semblance: float = 0.0

    tree: Octree | None = None
    parent: Node | None = None
    children: tuple[Node, ...] = ()

    _hash: bytes | None = None
    _children_cached: tuple[Node, ...] = ()
    _location: Location | None = None

    def split(self) -> tuple[Node, ...]:
        """Split the node into 8 children."""
        if not self.tree:
            raise EnvironmentError("Parent octree is not set.")

        if not self.can_split():
            raise NodeSplitError("Cannot split node below limit.")

        if not self._children_cached:
            child_size = self.size / 2

            self._children_cached = tuple(
                Node(
                    east=self.east + east * child_size / 2,
                    north=self.north + north * child_size / 2,
                    depth=self.depth + depth * child_size / 2,
                    size=child_size,
                    tree=self.tree,
                    parent=self,
                    level=self.level + 1,
                )
                for east in (-1, 1)
                for north in (-1, 1)
                for depth in (-1, 1)
            )

        self.children = self._children_cached
        self.tree._add_nodes(self.children)

        return self.children

    @property
    def coordinates(self) -> tuple[float, float, float]:
        """Returns the node coordinates (east, north, depth)."""
        return self.east, self.north, self.depth

    @property
    def radius(self) -> float:
        """Returns the radius of the sphere that can fit the node inside."""
        return np.sqrt(3 * (self.size / 2) ** 2)

    def get_distance_border(self, with_surface: bool = False) -> float:
        """Distance to the closest EW, NS or bottom border of the tree.

        Args:
            with_surface (bool, optional): If True, the distance to the closest border
                within the surface (open top) is returned. Defaults to False.

        Raises:
            AttributeError: If the parent tree is not set.

        Returns:
            float: Distance to the closest border.
        """
        if not self.tree:
            raise AttributeError("parent octree not set")
        tree = self.tree
        border_distance = min(
            self.north - tree.north_bounds[0],
            tree.north_bounds[1] - self.north,
            self.east - tree.east_bounds[0],
            tree.east_bounds[1] - self.east,
            tree.depth_bounds[1] - self.depth,
        )
        if with_surface:
            return min(border_distance, self.depth - tree.depth_bounds[0])
        return border_distance

    def is_inside_border(
        self,
        with_surface: bool = False,
        border_width: float | Literal["root_node_size"] = "root_node_size",
    ) -> bool:
        """Check if the node is within the root node border.

        Args:
            with_surface (bool, optional): If True, the surface is considered
                as a border. Defaults to False.
            border_width (float, optional): Width of the border, if 0.0,
                the octree's root node size is used. Defaults to 0.0.

        Returns:
            bool: True if the node is inside the root tree's border.
        """
        if self.tree is None:
            raise AttributeError("parent tree not set")
        border = (
            self.tree.root_node_size
            if border_width == "root_node_size"
            else border_width
        )
        return self.get_distance_border(with_surface) <= border

    def can_split(self) -> bool:
        """Check if the node can be split.

        Raises:
            AttributeError: If the parent tree is not set.

        Returns:
            bool: True if the node can be split.
        """
        if self.tree is None:
            raise AttributeError("parent octree not set")
        return (self.level + 1) < self.tree.n_levels

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node.
        """
        return not bool(self.children)

    def reset(self) -> None:
        """Reset the node to its initial state."""
        _ = [child.reset() for child in self.children]
        self.children = ()
        self.semblance = 0.0

    def set_tree(self, tree: Octree) -> None:
        """Set the parent tree of the node.

        Args:
            tree (Octree): The parent tree.
        """
        self.tree = tree
        for child in self.children:
            child.set_tree(tree)

    def distance_to_location(self, location: Location) -> float:
        """Three dimensional distance to a location.

        Args:
            location (Location): Location to calculate distance to.

        Returns:
            float: Distance to location.
        """
        return location.distance_to(self.as_location())

    def semblance_density(self) -> float:
        """Calculate the semblance density of the octree.

        Returns:
            The semblance density of the octree.
        """
        return self.semblance / self.size**3

    def as_location(self) -> Location:
        """Returns the location of the node.

        Returns:
            Location: Location of the node.
        """
        if not self.tree:
            raise AttributeError("parent octree not set")
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

    def is_colliding(self, other: Node) -> bool:
        """Check if two nodes collide.

        Args:
            other (Node): Other node to check for collision.

        Returns:
            bool: True if the nodes collide.
        """
        return (
            abs(self.east - other.east) <= (self.size + other.size) / 2
            and abs(self.north - other.north) <= (self.size + other.size) / 2
            and abs(self.depth - other.depth) <= (self.size + other.size) / 2
        )

    def get_neighbours(self, leafs_only: bool = True) -> list[Node]:
        """Get the direct neighbours of the node from the parent tree.

        Neighbours are nodes that touch by vertice, edge or face, regardless of the
        nodes level.

        Returns:
            list[Node]: List of direct neighbours.
            leafs_only (bool): If True, only leaf nodes are returned.
        """
        if not self.tree:
            raise AttributeError("parent octree not set")

        nodes = self.tree.leaf_nodes if leafs_only else self.tree.nodes

        return [node for node in nodes if self.is_colliding(node) and node is not self]

    def distance_to(self, other: Node) -> float:
        """Distance to another node.

        Args:
            other (Node): Other node to calculate distance to.

        Returns:
            float: Distance to other node.
        """
        return np.sqrt(
            (self.east - other.east) ** 2
            + (self.north - other.north) ** 2
            + (self.depth - other.depth) ** 2
        )

    def __iter__(self) -> Iterator[Node]:
        if self.children:
            for child in self.children:
                yield from child
        else:
            yield self

    def hash(self) -> bytes:
        if not self.tree:
            raise AttributeError("parent octree not set")
        if not self._hash:
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


class Octree(BaseModel, Iterator[Node]):
    location: Location = Field(
        default=Location(lat=0.0, lon=0.0),
        description="The geographical center of the octree.",
    )
    root_node_size: PositiveFloat = Field(
        default=1 * KM,
        description="Size of the root node at the initial level (level 0) in meters.",
    )
    n_levels: int = Field(
        default=5,
        ge=1,
        description="Number of levels in the octree, defining the final resolution"
        " of the detection. Default is 5.",
    )
    east_bounds: Range = Field(
        default=Range(-10 * KM, 10 * KM),
        description="East bounds of the octree in meters.",
    )
    north_bounds: Range = Field(
        default=Range(-10 * KM, 10 * KM),
        description="North bounds of the octree in meters.",
    )
    depth_bounds: Range = Field(
        default=Range(0 * KM, 20 * KM),
        description="Depth bounds of the octree in meters.",
    )

    _root_nodes: list[Node] = PrivateAttr([])
    _semblance: np.ndarray | None = PrivateAttr(None)
    _cached_coordinates: dict[CoordSystem, np.ndarray] = PrivateAttr({})
    _nodes: list[Node] = PrivateAttr([])

    model_config = ConfigDict(ignored_types=(cached_property,))

    @field_validator("location")
    def check_reference(cls, location: Location) -> Location:  # noqa: N805
        if location.lat == 0.0 and location.lon == 0.0:
            raise ValueError("invalid  location, expected non-zero lat/lon")
        return location

    @model_validator(mode="after")
    def check_limits(self) -> Octree:
        """Check that the size limits are valid."""
        for dimension, ext in zip(
            ("east", "north", "depth"), self.extent(), strict=True
        ):
            if ext % self.root_node_size:
                raise ValueError(
                    f"invalid octree root node size {self.root_node_size}:"
                    f" Not a multiple of the volume in {dimension} with size {ext}"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize octree. This method is called by the pydantic model."""
        self._root_nodes = self.create_root_nodes(self.root_node_size)
        self._add_nodes(self._root_nodes)

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
            self.east_bounds.max - self.east_bounds.min,
            self.north_bounds.max - self.north_bounds.min,
            self.depth_bounds.max - self.depth_bounds.min,
        )

    def create_root_nodes(self, length: float) -> list[Node]:
        ln = length
        ext_east, ext_north, ext_depth = self.extent()
        east_nodes = np.arange(ext_east // ln) * ln + ln / 2 + self.east_bounds.min
        north_nodes = np.arange(ext_north // ln) * ln + ln / 2 + self.north_bounds.min
        depth_nodes = np.arange(ext_depth // ln) * ln + ln / 2 + self.depth_bounds.min

        return [
            Node(east=east, north=north, depth=depth, size=ln, tree=self, level=0)
            for depth in depth_nodes
            for north in north_nodes
            for east in east_nodes
        ]

    @cached_property
    def n_nodes(self) -> int:
        """Number of nodes in the octree."""
        return len(self.nodes)

    @cached_property
    def n_leaf_nodes(self) -> int:
        """Number of nodes in the octree."""
        return len(self.leaf_nodes)

    @property
    def nodes(self) -> list[Node]:
        """List of nodes in the octree."""
        return self._nodes

    def _add_nodes(self, nodes: Iterable[Node]) -> None:
        self._nodes.extend(nodes)
        self._clear_cache()

    @property
    def volume(self) -> float:
        """Volume of the octree in cubic meters."""
        return reduce(mul, self.extent())

    @cached_property
    def leaf_nodes(self) -> list[Node]:
        """Get all leaf nodes of the octree.

        Returns:
            list[Node]: List of leaf nodes.
        """
        return [node for node in self if node.is_leaf()]

    def __iter__(self) -> Iterator[Node]:
        yield from self.nodes

    def __next__(self) -> Node:
        return next(iter(self.nodes))

    def __getitem__(self, idx: int) -> Node:
        try:
            return self.nodes[idx]
        except IndexError:
            raise IndexError(f"Bad node index {idx}") from None

    def _clear_cache(self) -> None:
        with contextlib.suppress(AttributeError):
            del self.n_nodes
        with contextlib.suppress(AttributeError):
            del self.n_leaf_nodes
        with contextlib.suppress(AttributeError):
            del self.leaf_nodes
        self._cached_coordinates.clear()
        self._semblance = None

    def reset(self) -> Self:
        """Reset the octree to its initial state and return it."""
        _ = [node.reset() for node in self._root_nodes]
        self._nodes.clear()
        self._add_nodes(self._root_nodes)
        return self

    def clear(self) -> None:
        """Clear the octree's cached data."""
        logger.debug("clearing octree")
        self._root_nodes = self.create_root_nodes(self.root_node_size)
        self._clear_cache()

    def set_level(self, level: int) -> None:
        """Set the octree to a specific level.

        Args:
            level (int): Level to set the octree to.
        """
        if not 0 <= level <= self.n_levels:
            raise ValueError(
                f"invalid level {level}, expected level <= {self.n_levels}"
            )
        logger.debug("setting tree to level %d", level)
        self.reset()
        for _ in range(level):
            for node in self.nodes.copy():
                node.split()

    def reduce_axis(
        self,
        surface: Literal["NE", "ED", "ND"] = "NE",
        max_level: int = -1,
        accumulator: Callable[np.ndarray] = np.max,
    ) -> np.ndarray:
        """Reduce the octree's nodes to the surface.

        Args:
            surface (Literal["NE", "ED", "ND"], optional): Surface to reduce to.
                Defaults to "NE".
            max_level (int, optional): Maximum level to reduce to. Defaults to -1.
            accumulator (Callable, optional): Accumulator function. Defaults to np.max.

        Returns:
            np.ndarray: Of shape (n-nodes, 4) with columns (east, north, depth, value).
        """
        groups = defaultdict(list)

        component_map = {
            "NE": lambda n: (n.east, n.north, n.size),
            "ED": lambda n: (n.east, n.depth, n.size),
            "ND": lambda n: (n.north, n.depth, n.size),
        }

        if surface not in component_map:
            raise ValueError(
                f"Unknown surface component: {surface}, expected NE, ED or ND."
            )

        for node in self.leaf_nodes:
            if max_level >= 0 and node.level > max_level:
                continue
            groups[component_map[surface](node)].append(node.semblance)

        semblances = (accumulator(values) for values in groups.values())
        return np.array(
            [(*node, val) for node, val in zip(groups.keys(), semblances, strict=True)]
        )

    @property
    def semblance(self) -> np.ndarray:
        """Returns the semblance values of all nodes."""
        return np.array([node.semblance for node in self])

    def map_semblance(self, semblance: np.ndarray, leaf_only: bool = True) -> None:
        """Maps semblance values to nodes.

        Args:
            semblance (np.ndarray): Of shape (n-nodes,).
            leaf_only (bool, optional): If True, only leaf nodes are mapped.
                Defaults to True.
        """
        self._semblance = semblance
        nodes = self.leaf_nodes if leaf_only else self.nodes

        for node, node_semblance in zip(nodes, semblance, strict=True):
            node.semblance = float(node_semblance)

    def get_coordinates(self, system: CoordSystem = "geographic") -> np.ndarray:
        if self._cached_coordinates.get(system) is None:
            coords = get_node_coordinates(self.nodes, system=system)
            coords.setflags(write=False)
            self._cached_coordinates[system] = coords
        return self._cached_coordinates[system]

    def distances_stations(self, stations: Stations) -> np.ndarray:
        """Returns the 3D distances from all nodes to all stations.

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

    def distances_stations_surface(self, stations: Stations) -> np.ndarray:
        """Returns the surface distance from all nodes to all stations.

        Args:
            stations (Stations): Stations to calculate distance to.

        Returns:
            np.ndarray: Distances in shape (n-nodes, n-stations).
        """
        node_coords = self.get_coordinates(system="geographic")
        n_nodes = node_coords.shape[0]

        node_coords = np.repeat(node_coords, stations.n_stations, axis=0)
        sta_coords = np.vstack(
            n_nodes * [stations.get_coordinates(system="geographic")]
        )
        return od.distance_accurate50m_numpy(
            node_coords[:, 0], node_coords[:, 1], sta_coords[:, 0], sta_coords[:, 1]
        ).reshape(-1, stations.n_stations)

    def get_nodes(self, indices: Iterable[int]) -> list[Node]:
        """Retrieves a list of nodes from the octree based on the given indices.

        Args:
            indices (Iterable[int]): The indices of the nodes to retrieve.

        Returns:
            list[Node]: A list of nodes corresponding to the given indices.
        """
        return [self[idx] for idx in indices]

    def get_nodes_by_threshold(self, semblance_threshold: float = 0.0) -> list[Node]:
        """Get all nodes with a semblance above a threshold.

        Args:
            semblance_threshold (float): Semblance threshold. Default is 0.0.

        Returns:
            list[Node]: List of nodes.
        """
        if not semblance_threshold:
            return list(self)
        return [node for node in self if node.semblance >= semblance_threshold]

    def get_nodes_level(self, level: int = 0) -> list[Node]:
        """Get all nodes at a specific level.

        Args:
            level (int): Level to get nodes from.

        Returns:
            list[Node]: List of nodes.
        """
        return [node for node in self if node.level <= level]

    def smallest_node_size(self) -> float:
        """Returns the smallest possible node size.

        Returns:
            float: Smallest possible node size.
        """
        return self.root_node_size / (2 ** (self.n_levels - 1))

    def total_number_nodes(self) -> int:
        """Returns the total number of nodes of all levels.

        Returns:
            int: Total number of nodes.
        """
        return len(self._root_nodes) * (8 ** (self.n_levels - 1))

    async def interpolate_max_semblance(
        self,
        peak_node: Node,
    ) -> Location:
        """Interpolate the location of the maximum semblance value.

        This method calculates the location of the maximum semblance value by performing
        interpolation using surrounding nodes. It uses the scipy Rbf (Radial basis function)
        interpolation method to fit a smooth function to the given data points. The function
        is then minimized to find the location of the maximum value.

        Returns:
            Location: Location of the maximum semblance value.

        Raises:
            AttributeError: If no semblance values are set.
        """
        if self._semblance is None:
            raise AttributeError("no semblance values set")

        neighbor_nodes = peak_node.get_neighbours()
        neighbor_coords = np.array(
            [
                (node.east, node.north, node.depth, node.semblance)
                for node in [peak_node, *neighbor_nodes]
            ]
        )

        neighbor_semblance = neighbor_coords[:, 3]
        rbf = scipy.interpolate.RBFInterpolator(
            neighbor_coords[:, :3],
            neighbor_semblance,
            kernel="cubic",
        )
        bound = peak_node.size / 1.5
        res = await asyncio.to_thread(
            scipy.optimize.minimize,
            lambda x: -rbf(np.atleast_2d(x)),
            method="Nelder-Mead",
            bounds=(
                (peak_node.east - bound, peak_node.east + bound),
                (peak_node.north - bound, peak_node.north + bound),
                (peak_node.depth - bound, peak_node.depth + bound),
            ),
            x0=(peak_node.east, peak_node.north, peak_node.depth),
        )

        reference = self.location
        result = Location(
            lat=reference.lat,
            lon=reference.lon,
            elevation=reference.elevation,
            east_shift=reference.east_shift + res.x[0],
            north_shift=reference.north_shift + res.x[1],
            depth=reference.depth + res.x[2],
        )
        logger.info(
            "interpolated source offset (e-n-d): %.1f, %.1f, %.1f",
            *peak_node.as_location().offset_from(result),
        )
        return result

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
            node.set_tree(tree)
        return tree

    def save_pickle(self, filename: Path) -> None:
        """Save the octree to a pickle file.

        Args:
            filename (Path): Filename to save to.
        """
        import pickle

        logger.info("saving octree pickle to %s", filename)
        with filename.open("wb") as f:
            pickle.dump(self, f)

    def get_corners(self) -> list[Location]:
        """Get the corners of the octree.

        Returns:
            list[Location]: List of locations.
        """
        reference = self.location
        return [
            Location(
                lat=reference.lat,
                lon=reference.lon,
                elevation=reference.elevation,
                east_shift=reference.east_shift + east,
                north_shift=reference.north_shift + north,
                depth=reference.depth + depth,
            )
            for east in (self.east_bounds.min, self.east_bounds.max)
            for north in (self.north_bounds.min, self.north_bounds.max)
            for depth in (self.depth_bounds.min, self.depth_bounds.max)
        ]

    def __hash__(self) -> int:
        return hash(
            (
                self.root_node_size,
                self.n_levels,
                self.east_bounds,
                self.north_bounds,
                self.depth_bounds,
            )
        )
