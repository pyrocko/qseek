from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FFMpegFileWriter, FuncAnimation
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from qseek.plot.base import BasePlot, LassieFigure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap

    from qseek.octree import Octree

logger = logging.getLogger(__name__)


def octree_to_rectangles(
    octree: Octree,
    cmap: str | Colormap = "magma_r",
    normalize: bool = False,
) -> PatchCollection:
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    coords = octree.reduce_surface()
    coords = coords[np.argsort(coords[:, 2])[::-1]]
    size_order = np.argsort(coords[:, 2])[::-1]
    coords = coords[size_order]

    sizes = coords[:, 2]
    semblances = coords[:, 3]
    sizes = sorted(set(sizes), reverse=True)
    # zorders = {size: 1.0 + float(order) for order, size in enumerate(sizes)}

    rectangles = []
    for node in coords:
        east, north, size, semblance = node
        half_size = size / 2
        rect = Rectangle(
            xy=(east - half_size, north - half_size),
            width=size,
            height=size,
        )
        rectangles.append(rect)

    if normalize:
        semblances /= semblances.max()
    colors = cmap(semblances)
    edge_colors = cm.get_cmap("binary")(semblances**2, alpha=0.8)

    return PatchCollection(
        patches=rectangles,
        facecolors=colors,
        edgecolors=edge_colors,
        linewidths=0.1,
    )


class OctreeRefinement(BasePlot):
    normalize: bool = False
    plot_detections: bool = False

    def get_figure(self) -> Iterator[LassieFigure]:
        yield self.create_figure()

    def create_figure(self) -> LassieFigure:
        figure = self.new_figure("octree-refinement.png")
        ax = figure.get_axes()
        octree = self.search.octree

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.add_collection(octree_to_rectangles(octree, normalize=self.normalize))

        ax.set_title(f"Octree surface tiles (nodes: {octree.n_nodes})")

        ax.autoscale()

        if self.plot_detections:
            detections = self.search.detections
            for detection in detections or []:
                ax.scatter(
                    detection.east_shift,
                    detection.north_shift,
                    marker="*",
                    s=50,
                    color="yellow",
                )
        return figure


def plot_octree_3d(octree: Octree, cmap: str = "Oranges") -> None:
    ax = plt.figure().add_subplot(projection="3d")
    colormap = get_cmap(cmap)

    coords = octree.get_coordinates("cartesian").T
    colors = colormap(octree.semblance, alpha=octree.semblance)

    ax.scatter(coords[0], coords[1], coords[2], c=colors)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Depth [m]")
    plt.show()


def plot_octree_scatter(
    octree: Octree,
    accumulator: Callable = np.max,
    cmap: str = "magma_r",
) -> None:
    colormap = get_cmap(cmap)

    surface = octree.reduce_surface(accumulator)
    normalized_semblance = octree.semblance / octree.semblance.max()

    colors = colormap(surface[:, 2], alpha=normalized_semblance)
    ax = plt.figure().gca()
    ax.scatter(surface[:, 0], surface[:, 1], c=colors)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    plt.show()


def plot_octree_semblance_movie(
    octree: Octree,
    semblance: np.ndarray,
    file: Path,
    cmap: str = "magma_r",
) -> None:
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection="3d")
    colormap = get_cmap(cmap)

    coords = octree.get_coordinates("cartesian").T
    nodes = ax.scatter(coords[0], coords[1], coords[2])

    def update(frame_number: int) -> None:
        slice_semblance = semblance[:, frame_number]
        normalized_semblance = slice_semblance / slice_semblance.max()
        colors = colormap(slice_semblance, alpha=normalized_semblance)
        nodes.set_color(colors)

    n_frames = semblance.shape[1]
    animation = FuncAnimation(fig, update, frames=n_frames)
    writer = FFMpegFileWriter(fps=30)
    logger.info("saving semblance movie to %s (n_frames=%d)", file, n_frames)
    animation.save(str(file), writer=writer)
