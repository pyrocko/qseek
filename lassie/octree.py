from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, ClassVar, Iterator

import numpy as np
from pydantic import BaseModel, PrivateAttr

from lassie.models.location import Location

if TYPE_CHECKING:
    from lassie.models.receiver import Receiver

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
        self.children = tuple()
        self._children_cached = self.children
        self.semblance = 0.0

    def distance_receiver(self, receiver: Receiver) -> float:
        return receiver.distance_to(self.as_location())

    def as_location(self) -> Location:
        if not self._tree:
            raise ValueError("Parent tree not set.")
        return Location(
            lat=self._tree.center_lat,
            lon=self._tree.center_lon,
            elevation=self._tree.surface_elevation,
            east_shift=self.east,
            north_shift=self.north,
        )

    def __iter__(self) -> Iterator[Node]:
        if self.children:
            for child in self.children:
                yield from child
        else:
            yield self

    def __hash__(self) -> int:
        return hash((self.east, self.north, self.depth, self.size))


class Octree(BaseModel):
    _root_nodes: list[Node] = PrivateAttr([])
    center_lat: float = 0.0
    center_lon: float = 0.0
    surface_elevation: float = 0.0
    size_initial: float = 2 * km
    size_limit: float = 500
    east_bounds: tuple[float, float] = (-10 * km, 10 * km)
    north_bounds: tuple[float, float] = (-10 * km, 10 * km)
    depth_bounds: tuple[float, float] = (0 * km, 20 * km)
    nodes: list[Node] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.init_nodes()

    def extent(self) -> tuple[float, float, float]:
        return (
            self.east_bounds[1] - self.east_bounds[0],
            self.north_bounds[1] - self.north_bounds[0],
            self.depth_bounds[1] - self.depth_bounds[0] + self.surface_elevation,
        )

    def init_nodes(self) -> None:
        logger.debug("initializing nodes")
        ext = self.size_initial
        ext_east, ext_north, ext_depth = self.extent()
        east_nodes = np.arange(ext_east // ext) * ext + ext / 2 + self.east_bounds[0]
        north_nodes = np.arange(ext_north // ext) * ext + ext / 2 + self.north_bounds[0]
        depth_nodes = (
            np.arange(ext_depth // ext) * ext
            + ext / 2
            + self.depth_bounds[0]
            + self.surface_elevation
        )

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
    def nnodes(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[Node]:
        for node in self._root_nodes:
            yield from node

    def __getitem__(self, idx: int) -> Node:
        for inode, node in enumerate(self):
            if inode == idx:
                return node
        raise IndexError(f"Unknown node {idx}")

    def reset(self) -> None:
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
