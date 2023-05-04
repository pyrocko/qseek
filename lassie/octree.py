from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, ClassVar, Iterator

import numpy as np
from pydantic import BaseModel, PositiveFloat, PrivateAttr, validator

from lassie.models.location import Location

if TYPE_CHECKING:
    from lassie.models.station import Station

logger = logging.getLogger(__name__)
km = 1e3


class Node(BaseModel):
    east: float
    north: float
    depth: float
    size: float
    semblance: float = 0.0

    children: tuple[Node] = tuple()

    _children_cached: tuple[Node] = PrivateAttr(tuple())
    _tree: ClassVar[Octree | None] = None
    _location: Location | None = PrivateAttr(None)

    def split(self) -> tuple[Node]:
        if not self._tree:
            raise EnvironmentError("Parent tree is not set.")

        if not self._children_cached:
            half_size = self.size / 2
            if half_size < self._tree.size_limit:
                raise ValueError("Cannot split node below limit.")

            self._children_cached = tuple(
                Node.construct(
                    east=self.east + east * half_size / 2,
                    north=self.north + north * half_size / 2,
                    depth=self.depth + depth * half_size / 2,
                    size=half_size,
                )
                for east in (-1, 1)
                for north in (-1, 1)
                for depth in (-1, 1)
            )

        self.children = self._children_cached
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
        if not self._tree:
            raise AttributeError("parent tree not set")
        if not self._location:
            self._location = Location.construct(
                lat=self._tree.center_lat,
                lon=self._tree.center_lon,
                elevation=self._tree.surface_elevation,
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
        if not self._tree:
            raise AttributeError("parent tree not set")
        return hash(
            (
                self._tree.center_lat,
                self._tree.center_lon,
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

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.init_nodes()

    @validator("east_bounds", "north_bounds", "depth_bounds")
    def _check_bounds(cls, bounds):  # noqa: N805
        for value in bounds:
            if value[0] >= value[1]:
                raise ValueError(f"invalid bounds {value}")
        return bounds

    @validator("size_initial", "size_limit")
    def _check_limits(cls, limits):  # noqa: N805
        if limits[0] < limits[1]:
            raise ValueError(f"invalid octree size limits {limits}")
        return limits

    def extent(self) -> tuple[float, float, float]:
        return (
            self.east_bounds[1] - self.east_bounds[0],
            self.north_bounds[1] - self.north_bounds[0],
            self.depth_bounds[1] - self.depth_bounds[0],
        )

    def init_nodes(self) -> None:
        logger.debug("initializing nodes")
        ext = self.size_initial
        ext_east, ext_north, ext_depth = self.extent()
        east_nodes = np.arange(ext_east // ext) * ext + ext / 2 + self.east_bounds[0]
        north_nodes = np.arange(ext_north // ext) * ext + ext / 2 + self.north_bounds[0]
        depth_nodes = np.arange(ext_depth // ext) * ext + ext / 2 + self.depth_bounds[0]

        Node._tree = self
        self._root_nodes = [
            Node.construct(
                east=east,
                north=north,
                depth=depth,
                size=ext,
            )
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

    def reset(self) -> None:
        logger.debug("resetting tree")
        for node in self._root_nodes:
            node.reset()

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

    def get_coordinates(self) -> np.ndarray:
        return np.array([node.coordinates for node in self])

    def refine(self, semblance_threshold: float) -> list[Node]:
        new_nodes = []
        for node in self:
            if node.semblance > semblance_threshold:
                new_nodes.extend(node.split())
        return new_nodes

    def finest_tree(self) -> Octree:
        tree = self.copy()
        tree.reset()
        while True:
            try:
                for node in tree:
                    node.split()
            except ValueError:
                break

        tree._root_nodes = [node for node in tree]  # Cut off parent nodes
        return tree

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = self.get_coordinates().T
        ax.scatter(coords[0], coords[1], coords[2], c=self.semblance)
        plt.show()

    def plot_surface(self) -> None:
        import matplotlib.pyplot as plt

        surface = self.reduce_surface()
        ax = plt.figure().gca()
        ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 2])
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
