from __future__ import annotations

import numpy as np
import pytest

from qseek.models.location import Location
from qseek.octree import Node, NodeSplitError, Octree
from qseek.utils import Range

km = 1e3


def test_octree(octree: Octree, plot: bool) -> None:
    octree.reset()
    assert octree.n_nodes > 0

    nnodes = octree.n_nodes

    for node in octree.nodes.copy():
        node.split()

    assert nnodes * 8 == octree.n_leaf_nodes

    child, *_ = octree[80].split()
    while True:
        try:
            child, *_ = child.split()
        except NodeSplitError:
            break

    for node in octree:
        node.semblance = node.depth + node.east + node.north

    if plot:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = octree.get_coordinates().T
        ax.scatter(coords[0], coords[1], coords[2], c=octree.semblance)
        plt.show()

    surface = octree.reduce_axis()
    if plot:
        import matplotlib.pyplot as plt

        ax = plt.figure().gca()
        ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 2])
        plt.show()


@pytest.fixture
def refined_octree() -> Octree:
    """A fresh octree with leaf nodes of three sizes."""
    octree = Octree(
        location=Location(lat=10.0, lon=10.0, elevation=1.0 * km),
        root_node_size=2 * km,
        n_levels=3,
        east_bounds=Range(-10 * km, 10 * km),
        north_bounds=Range(-10 * km, 10 * km),
        depth_bounds=Range(0 * km, 10 * km),
    )
    octree.reset()
    for node in octree.nodes[::7].copy():
        node.split()
    for node in octree.leaf_nodes[::11].copy():
        if node.can_split():
            node.split()
    assert len({node.size for node in octree.leaf_nodes}) == 3
    return octree


def expected_neighbours(node: Node, leafs_only: bool = True) -> list[Node]:
    """The neighbours as found by the scan over all nodes."""
    nodes = node.tree.leaf_nodes if leafs_only else node.tree.nodes
    neighbours = list(filter(node.is_colliding, nodes))
    neighbours.remove(node)
    return neighbours


def node_ids(nodes: list[Node]) -> list[int]:
    return [id(node) for node in nodes]


def expected_densest(octree: Octree, semblance: np.ndarray) -> Node:
    """The densest leaf node as found by the scan over all leaf nodes."""
    octree.map_semblance(semblance, leaf_only=True)
    return max(octree.leaf_nodes, key=lambda n: n.semblance_density())


def test_get_neighbours(refined_octree: Octree) -> None:
    octree = refined_octree
    for leafs_only in (True, False):
        for node in octree.leaf_nodes[::5]:
            neighbours = node.get_neighbours(leafs_only=leafs_only)
            assert node_ids(neighbours) == node_ids(
                expected_neighbours(node, leafs_only)
            )


def test_get_neighbours_of_split_node(refined_octree: Octree) -> None:
    node = next(node for node in refined_octree.nodes if node.children)
    # The node is not a leaf node and not among its leaf neighbours
    with pytest.raises(ValueError):
        node.get_neighbours()

    neighbours = node.get_neighbours(leafs_only=False)
    assert node_ids(neighbours) == node_ids(expected_neighbours(node, False))


def test_get_neighbours_cache(refined_octree: Octree) -> None:
    octree = refined_octree
    node = octree.leaf_nodes[0]
    before = node.get_neighbours()

    # Splitting a neighbour replaces it by its children
    neighbour = next(n for n in before if n.can_split())
    children = neighbour.split()
    after = node.get_neighbours()
    assert node_ids(after) == node_ids(expected_neighbours(node))
    assert neighbour not in after
    assert any(child in after for child in children)

    octree.reset()
    assert octree.get_node_geometry().shape == (5, octree.n_leaf_nodes)
    node = octree.leaf_nodes[0]
    assert node_ids(node.get_neighbours()) == node_ids(expected_neighbours(node))

    octree.set_level(1)
    assert octree.get_node_geometry().shape == (5, octree.n_leaf_nodes)
    assert octree.get_node_geometry(leafs_only=False).shape == (5, octree.n_nodes)
    node = octree.leaf_nodes[0]
    assert node_ids(node.get_neighbours()) == node_ids(expected_neighbours(node))


def test_get_densest_leaf_node(refined_octree: Octree) -> None:
    octree = refined_octree
    sizes = np.array([node.size for node in octree.leaf_nodes])
    smallest = sizes == sizes.min()
    rng = np.random.default_rng(1)
    for _ in range(20):
        # Negative and positive semblance of nodes of different sizes
        semblance = rng.normal(size=octree.n_leaf_nodes).astype(np.float32)
        densest = octree.get_densest_leaf_node(semblance)
        assert densest is expected_densest(octree, semblance)

        # Ties: all smallest nodes have the maximum density, the first one wins
        semblance = rng.random(octree.n_leaf_nodes).astype(np.float32)
        semblance[smallest] = semblance.max()
        densest = octree.get_densest_leaf_node(semblance)
        assert densest is expected_densest(octree, semblance)
        assert densest is octree.leaf_nodes[int(np.argmax(smallest))]


def test_get_densest_leaf_node_nan(refined_octree: Octree) -> None:
    octree = refined_octree
    semblance = np.ones(octree.n_leaf_nodes, dtype=np.float32)
    semblance[[3, 7]] = np.nan
    assert octree.get_densest_leaf_node(semblance) is octree.leaf_nodes[3]


@pytest.mark.parametrize("deep", [False, True])
def test_model_copy(refined_octree: Octree, deep: bool) -> None:
    octree = refined_octree
    copy = octree.model_copy(deep=deep)
    assert copy.n_nodes == octree.n_nodes
    assert all(node.tree is copy for node in copy._root_nodes)
    if deep:
        assert all(node.tree is copy for node in copy)
        n_nodes = octree.n_nodes
        copy.leaf_nodes[0].split()
        assert octree.n_nodes == n_nodes
        assert copy.n_nodes == n_nodes + 8


def test_cached_bottom(refined_octree: Octree) -> None:
    bottom = refined_octree.cached_bottom()
    assert bottom is not refined_octree
    assert bottom.n_nodes >= refined_octree.n_nodes
