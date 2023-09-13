from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FFMpegFileWriter, FuncAnimation
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from lassie.models.detection import EventDetection

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap

    from lassie.octree import Octree

logger = logging.getLogger(__name__)


def octree_to_rectangles(
    octree: Octree,
    cmap: str | Colormap = "Oranges",
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
    return PatchCollection(
        patches=rectangles,
        facecolors=colors,
        edgecolors=(0, 0, 0, 0.3),
        linewidths=0.5,
    )


def plot_octree_3d(octree: Octree, cmap: str = "Oranges") -> None:
    ax = plt.figure().add_subplot(projection="3d")
    colormap = get_cmap(cmap)

    coords = octree.get_coordinates("cartesian").T
    colors = colormap(octree.semblance, alpha=octree.semblance)

    ax.scatter(coords[0], coords[1], coords[2], c=colors)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_zlabel("depth [m]")
    plt.show()


def plot_octree_scatter(
    octree: Octree,
    accumulator: Callable = np.max,
    cmap: str = "Oranges",
) -> None:
    colormap = get_cmap(cmap)

    surface = octree.reduce_surface(accumulator)
    normalized_semblance = octree.semblance / octree.semblance.max()

    colors = colormap(surface[:, 2], alpha=normalized_semblance)
    ax = plt.figure().gca()
    ax.scatter(surface[:, 0], surface[:, 1], c=colors)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    plt.show()


def plot_octree_surface_tiles(
    octree: Octree,
    axes: plt.Axes | None = None,
    normalize: bool = False,
    filename: Path | None = None,
    detections: list[EventDetection] | None = None,
) -> None:
    if axes is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = axes.figure
        ax = axes

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.add_collection(octree_to_rectangles(octree, normalize=normalize))

    ax.set_title(f"Octree surface tiles (nodes: {octree.n_nodes})")

    ax.autoscale()
    for detection in detections or []:
        ax.scatter(
            detection.east_shift,
            detection.north_shift,
            marker="*",
            s=50,
            color="yellow",
        )
    if filename is not None:
        fig.savefig(str(filename), bbox_inches="tight", dpi=300)
        plt.close()
    elif axes is None:
        plt.show()


def plot_octree_semblance_movie(
    octree: Octree,
    semblance: np.ndarray,
    file: Path,
    cmap: str = "Oranges",
) -> None:
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection="3d")
    colormap = get_cmap("Oranges")

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
