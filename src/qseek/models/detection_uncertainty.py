from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Self

if TYPE_CHECKING:
    from qseek.octree import Node, Octree


# Equivalent to one standard deviation
THRESHOLD = 1.0 / np.sqrt(np.e)


class DetectionUncertainty(BaseModel):
    east_uncertainties: tuple[float, float] = Field(
        ...,
        description="Uncertainty in east direction in [m].",
    )
    north_uncertainties: tuple[float, float] = Field(
        ...,
        description="Uncertainty in north direction in [m].",
    )
    depth_uncertainties: tuple[float, float] = Field(
        ...,
        description="Uncertainty in depth in [m].",
    )

    @classmethod
    def from_event(
        cls, source_node: Node, octree: Octree, width: float = THRESHOLD
    ) -> Self:
        """
        Calculate the uncertainty of an event detection.

        Args:
            event: The event detection to calculate the uncertainty for.
            octree: The octree to use for the calculation.

        Returns:
            The calculated uncertainty.
        """
        nodes = octree.get_nodes(semblance_threshold=width)
        vicinity_coords = np.array(
            [(node.east, node.north, node.depth) for node in nodes]
        )
        relative_node_offsets = vicinity_coords - np.array(
            [source_node.east, source_node.north, source_node.depth]
        )
        min_offsets = np.min(relative_node_offsets, axis=0)
        max_offsets = np.max(relative_node_offsets, axis=0)

        return cls(
            east_uncertainties=(float(min_offsets[0]), float(max_offsets[0])),
            north_uncertainties=(float(min_offsets[1]), float(max_offsets[1])),
            depth_uncertainties=(float(min_offsets[2]), float(max_offsets[2])),
        )
