from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from lassie.models.location import Location
from lassie.tracers.cake import TravelTimeTree

if TYPE_CHECKING:
    from lassie.models.station import Stations
    from lassie.octree import Octree

KM = 1e3


def test_sptree_model(travel_time_tree: TravelTimeTree):
    model = travel_time_tree

    with TemporaryDirectory() as d:
        tmp = Path(d)
        file = model.save(tmp)

        model2 = TravelTimeTree.load(file)
        model2._load_sptree()

    source = Location(
        lat=0.0,
        lon=0.0,
        north_shift=1 * KM,
        east_shift=1 * KM,
        depth=5.0 * KM,
    )
    receiver = Location(
        lat=0.0,
        lon=0.0,
        north_shift=0 * KM,
        east_shift=0 * KM,
        depth=0,
    )

    model.get_travel_time(source, receiver)


def test_lut(
    travel_time_tree: TravelTimeTree,
    octree: Octree,
    stations: Stations,
) -> None:
    model = travel_time_tree
    model.init_lut(octree, stations)

    traveltimes_tree = model.interpolate_travel_times(octree, stations)
    traveltimes_lut = model.get_travel_times(octree, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)

    # Test refilling the LUT
    model._node_lut.clear()
    traveltimes_tree = model.interpolate_travel_times(octree, stations)
    traveltimes_lut = model.get_travel_times(octree, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)
    assert len(model._node_lut) > 0, "did not refill lut"

    stations_selection = stations.model_copy()
    stations_selection.stations = stations_selection.stations[:5]
    traveltimes_tree = model.interpolate_travel_times(octree, stations_selection)
    traveltimes_lut = model.get_travel_times(octree, stations_selection)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)
