from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel
from typing_extensions import Self

if TYPE_CHECKING:
    from qseek.octree import Node, Octree


# Equivalent to one standard deviation
THRESHOLD = 1.0 / np.sqrt(np.e)


class EventUncertainty(BaseModel):
    measure: Literal["standard_deviation"] = "standard_deviation"

    east_uncertainties: tuple[float, float]
    north_uncertainties: tuple[float, float]
    depth_uncertainties: tuple[float, float]

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
        eastings = vicinity_coords[:, 0] - source_node.east
        northings = vicinity_coords[:, 1] - source_node.north
        depths = vicinity_coords[:, 2] - source_node.depth
        return cls(
            east_uncertainties=(float(np.min(eastings)), float(np.max(eastings))),
            north_uncertainties=(float(np.min(northings)), float(np.max(northings))),
            depth_uncertainties=(float(np.min(depths)), float(np.max(depths))),
        )
