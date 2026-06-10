import asyncio

import numpy as np
from nicegui import Event
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from qseek.ui.state import CatalogStore

_CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_NOISE_COLOR = "#000000"


def labels_to_colors(labels: np.ndarray) -> list[str]:
    """Map DBSCAN labels to hex color strings; -1 (noise) gets black at low opacity."""
    return [
        _NOISE_COLOR if label == -1 else _CLUSTER_COLORS[label % len(_CLUSTER_COLORS)]
        for label in labels
    ]


async def get_distance_matrix(catalog: CatalogStore) -> np.ndarray:
    cartesian_coords = np.array(
        [catalog.east_shifts, catalog.north_shifts, catalog.depths]
    ).T
    return await asyncio.to_thread(cdist, cartesian_coords, cartesian_coords)


class ClusterDBScan:
    epsilon: float
    min_samples: int

    updated: Event

    def __init__(
        self,
        catalog: CatalogStore,
        epsilon: float = 3000,
        min_samples: int = 30,
    ) -> None:
        self.epsilon = epsilon
        self.min_samples = min_samples
        self._catalog = catalog

        self.updated = Event()

    async def get_labels(self) -> np.ndarray:
        distance_matrix = await get_distance_matrix(self._catalog)
        dbscan = DBSCAN(
            eps=self.epsilon,
            min_samples=self.min_samples,
            metric="precomputed",
        )

        return await asyncio.to_thread(dbscan.fit_predict, distance_matrix)
