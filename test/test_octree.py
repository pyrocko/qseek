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
        extent_initial=5 * km,
        extent_limit=0.2 * km,
    )


def test_octree(octree: Octree, plot: bool) -> None:
    assert octree.nnodes > 0

    nnodes = octree.nnodes

    for node in octree:
        node.split()

    assert nnodes * 8 == octree.nnodes

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


def test_finest_tree(octree: Octree, plot: bool):
    octree.finest_tree()


def test_refine(octree: Octree, plot: bool) -> None:
    for node in octree:
        if (
            node.depth > 30 * km
            and -10 * km < node.north < 10 * km
            and -10 * km < node.east < 10 * km
        ):
            node.semblance = 10.0

    new_nodes = octree.refine(semblance_threshold=2.0)
    assert new_nodes
    for node in new_nodes:
        node.semblance = 10.0

    if plot:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
        coords = octree.get_coordinates().T
        ax.scatter(coords[0], coords[1], coords[2], c=octree.semblance)
        plt.show()


def test_serialization(octree: Octree):
    octree.nodes = [node for node in octree]
    for node in octree:
        node.split()
    octree.json()
