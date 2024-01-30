from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Self

if TYPE_CHECKING:
    from qseek.octree import Node, Octree


# Equivalent to one standard deviation
PERCENTILE = 0.02


class DetectionUncertainty(BaseModel):
    east: tuple[float, float] = Field(
        ...,
        description="Uncertainty in east direction in [m].",
    )
    north: tuple[float, float] = Field(
        ...,
        description="Uncertainty in north direction in [m].",
    )
    depth: tuple[float, float] = Field(
        ...,
        description="Uncertainty in depth in [m].",
    )

    @classmethod
    def from_event(
        cls, source_node: Node, octree: Octree, percentile: float = PERCENTILE
    ) -> Self:
        """
        Calculate the uncertainty of an event detection.

        Args:
            event: The event detection to calculate the uncertainty for.
            octree: The octree to use for the calculation.
            percentile: The percentile to use for the calculation.
                Defaults to 0.02 (2%).

        Returns:
            The calculated uncertainty.
        """
        if not source_node.semblance:
            raise ValueError("Source node must have semblance value.")

        nodes = octree.get_nodes(
            semblance_threshold=source_node.semblance * (1.0 - percentile)
        )
        vicinity_coords = np.array(
            [(node.east, node.north, node.depth) for node in nodes]
        )
        relative_node_offsets = vicinity_coords - np.array(
            [source_node.east, source_node.north, source_node.depth]
        )
        min_offsets = np.min(relative_node_offsets, axis=0)
        max_offsets = np.max(relative_node_offsets, axis=0)

        return cls(
            east=(float(min_offsets[0]), float(max_offsets[0])),
            north=(float(min_offsets[1]), float(max_offsets[1])),
            depth=(float(min_offsets[2]), float(max_offsets[2])),
        )
