import pytest

from lassie.octree import Octree

km = 1e3


@pytest.fixture(scope="function")
def octree():
    yield Octree(
        center_lat=0.0,
        center_lon=0.0,
        east_bounds=(-25 * km, 25 * km),
        north_bounds=(-25 * km, 25 * km),
        depth_bounds=(0, 40 * km),
        size_initial=5 * km,
        size_limit=0.5 * km,
    )


def test_octree(octree: Octree, plot: bool) -> None:
    assert octree.n_nodes > 0

    nnodes = octree.n_nodes

    for node in octree:
        node.split()

    assert nnodes * 8 == octree.n_nodes

    child, *_ = octree[80].split()
    child, *_ = child.split()

    for node in octree:
        node.semblance = node.depth + node.east + node.north

    if plot:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = octree.get_coordinates().T
        ax.scatter(coords[0], coords[1], coords[2], c=octree.semblance)
        plt.show()

    surface = octree.reduce_surface()
    if plot:
        import matplotlib.pyplot as plt

        ax = plt.figure().gca()
        ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 2])
        plt.show()
