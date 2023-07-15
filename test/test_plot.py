from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from lassie.models.detection import EventDetection
from lassie.plot.octree import plot_octree_surface_tiles
from lassie.utils import datetime_now

if TYPE_CHECKING:
    from lassie.octree import Octree


@pytest.mark.plot
def test_octree_2d(octree: Octree) -> None:
    semblance = np.random.uniform(size=octree.n_nodes)
    octree.map_semblance(semblance)
    plot_octree_surface_tiles(octree, filename=Path("/tmp/test.png"))

    detection = EventDetection(
        lat=0.0,
        lon=0.0,
        east_shift=0.0,
        north_shift=0.0,
        distance_border=1000.0,
        semblance=1.0,
        time=datetime_now(),
    )

    fig = plt.figure()
    ax = fig.gca()

    plot_octree_surface_tiles(octree, axes=ax, detections=[detection])
    plt.show()
