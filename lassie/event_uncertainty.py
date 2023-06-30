from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from lassie.octree import Node, Octree


# Equivalent to one standard deviation
THRESHOLD = 1.0 / np.sqrt(np.e)


class EventUncertainty(BaseModel):
    measure: Literal["standard_deviation"] = "standard_deviation"

    east_uncertainties: tuple[float, float]
    north_uncertainties: tuple[float, float]
    depth_uncertainties: tuple[float, float]

    @classmethod
    def from_event(cls, source_node: Node, octree: Octree) -> Self:
        """
        Calculate the uncertainty of an event detection.

        Args:
            event: The event detection to calculate the uncertainty for.
            octree: The octree to use for the calculation.

        Returns:
            The calculated uncertainty.
        """
        nodes = octree.get_nodes(semblance_threshold=THRESHOLD)
        eastings = np.array([node.easting for node in nodes]) - source_node.easting
        northings = np.array([node.easting for node in nodes]) - source_node.northing
        depths = np.array([node.depth for node in nodes]) - source_node.depth
        return cls(
            east_uncertainties=(np.min(eastings), np.max(eastings)),
            north_uncertainties=(np.min(northings), np.max(northings)),
            depth_uncertainties=(np.min(depths), np.max(depths)),
        )
