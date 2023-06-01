from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegFileWriter, FuncAnimation
from matplotlib.cm import get_cmap

if TYPE_CHECKING:
    from lassie.octree import Octree

logger = logging.getLogger(__name__)


def plot_octree(octree: Octree, cmap: str = "Oranges") -> None:
    ax = plt.figure().add_subplot(projection="3d")
    colormap = get_cmap(cmap)

    coords = octree.get_coordinates("cartesian").T
    colors = colormap(octree.semblance, alpha=octree.semblance)

    ax.scatter(coords[0], coords[1], coords[2], c=colors)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_zlabel("depth [m]")
    plt.show()


def plot_octree_surface(
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


def plot_octree_movie(
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
