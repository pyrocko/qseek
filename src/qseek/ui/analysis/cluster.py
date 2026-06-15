from __future__ import annotations

import asyncio
import logging

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

logger = logging.getLogger(__name__)


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

    labels: np.ndarray | None = None
    colors: list[str] | None = None

    _distance_matrix: np.ndarray | None = None

    updated: Event

    def __init__(self, catalog: CatalogStore) -> None:
        self._catalog = catalog

        self.updated = Event()

    async def prepare(self):
        self._distance_matrix = await get_distance_matrix(self._catalog)

    async def cluster(
        self,
        epsilon: float = 3000,
        min_samples: int = 30,
    ) -> np.ndarray:
        if self._distance_matrix is None:
            await self.prepare()

        logger.info(
            "Clustering with DBSCAN (epsilon=%.1f, min_samples=%d)...",
            epsilon,
            min_samples,
        )
        dbscan = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric="precomputed",
        )

        self.labels = await asyncio.to_thread(dbscan.fit_predict, self._distance_matrix)
        self.colors = labels_to_colors(self.labels)
        return self.labels

    def unique_labels(self) -> set[int]:
        if self.labels is None:
            return set()
        return set(self.labels)

    def n_unique_labels(self) -> int:
        if self.labels is None:
            return 0
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)
