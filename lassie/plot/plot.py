from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from lassie.octree import Octree


def octree_to_rectangles(
    octree: Octree,
    cmap: str | Colormap = "Oranges",
) -> PatchCollection:
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    coords = octree.reduce_surface()
    coords = coords[np.argsort(coords[:, 2])[::-1]]
    sizes = coords[:, 2]
    semblances = coords[:, 3]
    sizes = sorted(set(sizes), reverse=True)
    zorders = {size: 1.0 + float(order) for order, size in enumerate(sizes)}
    print(zorders)

    rectangles = []
    for node in coords:
        east, north, size, semblance = node
        half_size = size / 2
        rect = Rectangle(
            xy=(east - half_size, north - half_size),
            width=size,
            height=size,
            zorder=semblance,
        )
        rectangles.append(rect)
    colors = cmap(semblances / semblances.max())
    print(colors)
    return PatchCollection(patches=rectangles, facecolors=colors, edgecolors="k")


def plot_octree(octree: Octree, axes: plt.Axes | None = None) -> None:
    if axes is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = axes
    ax.add_collection(octree_to_rectangles(octree))

    ax.autoscale()
    if axes is None:
        plt.show()
