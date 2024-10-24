from __future__ import annotations

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
