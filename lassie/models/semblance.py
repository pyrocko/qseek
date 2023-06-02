from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
from pydantic import BaseModel, PrivateAttr
from pyrocko import parstack
from pyrocko.trace import Trace
from scipy import signal, stats


class SemblanceStats(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    median_abs_deviation: float = 0.0

    _updates: int = PrivateAttr(0)

    def update(self, other: SemblanceStats) -> None:
        div = self._updates + 1
        self.mean = (self.mean * self._updates + other.mean) / div
        self.std = (self.std * self._updates + other.std) / div
        self.mad = (
            self.median_abs_deviation * self._updates + other.median_abs_deviation
        ) / div
        self._updates += 1


class Semblance:
    _max_semblance: np.ndarray | None = None
    _node_idx_max: np.ndarray | None = None

    def __init__(
        self,
        n_nodes: int,
        n_samples: int,
        start_time: datetime,
        sampling_rate: float,
        padding_samples: int = 0,
    ) -> None:
        self.start_time = start_time
        self.sampling_rate = sampling_rate
        self.padding_samples = padding_samples

        self._semblance_unpadded = np.zeros((n_nodes, n_samples), dtype=np.float32)

    @property
    def semblance(self) -> np.ndarray:
        if self.padding_samples:
            self.semblance = self._semblance_unpadded[
                :, self.padding_samples : -self.padding_samples
            ]
        else:
            self.semblance = self._semblance_unpadded

    @property
    def n_nodes(self) -> int:
        return self.semblance.shape[0]

    @property
    def n_samples(self) -> int:
        return self.semblance.shape[1]

    def maximum_semblance(self) -> np.ndarray:
        if not self._max_semblance:
            self._max_semblance = self.semblance.max(axis=0)
        return self._max_semblance

    async def maximum_node_idx(self, nparallel: int = 6) -> np.ndarray:
        if not self._node_idx_max:
            self._node_idx_max = await asyncio.to_thread(
                parstack.argmax,
                self.semblance.astype(np.float64),
                nparallel=nparallel,
            )
        return self._node_idx_max

    def mean(self) -> float:
        return float(self.maximum_semblance.mean())

    def std(self) -> float:
        return float(self.maximum_semblance.std())

    def median(self) -> float:
        return float(np.median(self.maximum_semblance))

    def median_mask(self, level: float = 3.0) -> np.ndarray:
        return self.maximum_semblance() < (self.median() * level)

    def median_abs_deviation(self) -> float:
        return float(stats.median_abs_deviation(self.maximum_semblance))

    def find_peaks(
        self,
        height: float,
        prominence: float,
        distance: float,
        trim_padding: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        detection_idx, _ = signal.find_peaks(
            self._semblance_unpadded.max(axis=0),
            height=height,
            prominence=prominence,
            distance=distance,
        )
        if trim_padding:
            detection_idx -= self.padding_samples
            detection_idx = detection_idx[detection_idx >= 0]
            detection_idx = detection_idx[detection_idx < self.semblance.size]
            semblance = self.semblance[detection_idx]
        else:
            semblance = self._semblance_unpadded[detection_idx]
        return detection_idx, semblance

    def get_stats(self) -> SemblanceStats:
        return SemblanceStats.from_orm(self)

    def get_trace(self) -> Trace:
        return Trace(
            network="",
            station="semblance",
            tmin=self.start_time.timestamp(),
            deltat=1.0 / self.sampling_rate,
            ydata=self.maximum_semblance(),
        )

    def _clear_cache(self) -> None:
        self._node_idx_max = None
        self._max_semblance = None

    def add(self, data: np.ndarray) -> None:
        """Stack samblance and normalize.

        Args:
            data (np.ndarray): Incoming semblance
        """
        self._semblance_unpadded += data
        self._clear_cache()

    def normalize(self, factor: int | float) -> None:
        self._semblance_unpadded /= factor
        self._clear_cache()
