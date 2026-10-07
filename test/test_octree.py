from __future__ import annotations

import numpy as np

from qseek.octree import NodeSplitError, Octree

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


def refined_octree(octree: Octree) -> Octree:
    octree.reset()
    for node in octree.nodes[::7].copy():
        node.split()
    for node in octree.leaf_nodes[::11].copy():
        if node.can_split():
            node.split()
    return octree


def test_get_neighbours(octree: Octree) -> None:
    octree = refined_octree(octree)
    for leafs_only in (True, False):
        nodes = octree.leaf_nodes if leafs_only else octree.nodes
        for node in octree.leaf_nodes[::5]:
            expected = list(filter(node.is_colliding, nodes))
            expected.remove(node)
            neighbours = node.get_neighbours(leafs_only=leafs_only)
            assert [id(n) for n in neighbours] == [id(n) for n in expected]

    # The cached geometry follows splits
    node = octree.leaf_nodes[0]
    n_neighbours = len(node.get_neighbours())
    octree.leaf_nodes[1].split()
    assert len(node.get_neighbours()) != n_neighbours


def test_get_densest_leaf_node(octree: Octree) -> None:
    octree = refined_octree(octree)
    rng = np.random.default_rng(1)
    for _ in range(20):
        semblance = rng.random(octree.n_leaf_nodes).astype(np.float32)
        # Ties: the first node wins
        semblance[rng.integers(0, semblance.size, 5)] = semblance.max()
        octree.map_semblance(semblance, leaf_only=True)
        expected = max(octree.leaf_nodes, key=lambda n: n.semblance_density())
        assert octree.get_densest_leaf_node(semblance) is expected
