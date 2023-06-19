from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, PrivateAttr
from pyrocko import parstack
from pyrocko.trace import Trace
from scipy import signal, stats

if TYPE_CHECKING:
    from datetime import datetime


class SemblanceStats(BaseModel):
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    median_abs_deviation: float = 0.0

    _updates: int = PrivateAttr(0)

    def update(self, other: SemblanceStats) -> None:
        updates = self._updates + 1

        def mean(old_attr, new_attr) -> float:
            return (old_attr * self._updates + new_attr) / updates

        self.median = mean(self.median, other.median)
        self.mean = mean(self.mean, other.mean)
        self.std = mean(self.std, other.std)
        self.median_abs_deviation = mean(
            self.median_abs_deviation, other.median_abs_deviation
        )
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
        self.n_samples_unpadded = n_samples

        self.semblance_unpadded = np.zeros((n_nodes, n_samples), dtype=np.float32)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.semblance.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.semblance.shape[1]

    @property
    def semblance(self) -> np.ndarray:
        padding_samples = self.padding_samples
        if padding_samples:
            return self.semblance_unpadded[:, padding_samples : -padding_samples - 1]
        return self.semblance_unpadded

    @property
    def maximum_semblance(self) -> np.ndarray:
        """Maximum semblance at any timestep in the volume.

        Returns:
            np.ndarray: Maximum semblance.
        """
        if self._max_semblance is None:
            self._max_semblance = self.semblance.max(axis=0)
        return self._max_semblance

    async def maximum_node_idx(self, nparallel: int = 6) -> np.ndarray:
        """Indices of maximum semblance at any time step.

        Args:
            nparallel (int, optional): Number of threads for calculation. Defaults to 6.

        Returns:
            np.ndarray: Node indices.
        """
        if self._node_idx_max is None:
            self._node_idx_max = await asyncio.to_thread(
                parstack.argmax,
                self.semblance.astype(np.float64),
                nparallel=nparallel,
            )
        return self._node_idx_max

    def mean(self) -> float:
        """Mean of the maximum semblance."""
        return float(self.maximum_semblance.mean())

    def std(self) -> float:
        """Standard deviation of the maximum semblance."""
        return float(self.maximum_semblance.std())

    def median(self) -> float:
        """Median of the maximum semblance."""
        return float(np.median(self.maximum_semblance))

    def median_abs_deviation(self) -> float:
        """Median absolute deviation of the maximum semblance."""
        return float(stats.median_abs_deviation(self.maximum_semblance))

    def median_mask(self, level: float = 3.0) -> np.ndarray:
        """Median mask above a level from the maximum semblance.

        This mask can be used for blinding peak amplitudes above a given level.

        Args:
            level (float, optional): Threshold level. Defaults to 3.0.

        Returns:
            np.ndarray: Boolean mask with peaks set to False.
        """
        return self.maximum_semblance < (self.median() * level)

    def find_peaks(
        self,
        height: float,
        prominence: float,
        distance: float,
        trim_padding: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find peaks in maximum semblance.

        For details see scipy.signal.find_peaks.

        Args:
            height (float): Minimum height of the peak.
            prominence (float): Prominence of the peak.
            distance (float): Minium distance of a peak to other peaks.
            trim_padding (bool, optional): Trim padded data in post-processing.
                Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices of peaks and peak values.
        """
        detection_idx, _ = signal.find_peaks(
            self.semblance_unpadded.max(axis=0),
            height=height,
            prominence=prominence,
            distance=distance,
        )
        if trim_padding:
            detection_idx -= self.padding_samples
            detection_idx = detection_idx[detection_idx >= 0]
            detection_idx = detection_idx[detection_idx < self.maximum_semblance.size]
            semblance = self.maximum_semblance[detection_idx]
        else:
            maximum_semblance = self.semblance_unpadded.max(axis=0)
            semblance = maximum_semblance[detection_idx]

        return detection_idx, semblance

    def get_stats(self) -> SemblanceStats:
        """Get statistics, like medianm, mean, std.

        Returns:
            SemblanceStats: Calculated statistics.
        """
        return SemblanceStats(
            median=self.median(),
            mean=self.mean(),
            std=self.std(),
            median_abs_deviation=self.median_abs_deviation(),
        )

    def get_trace(self, padded: bool = True) -> Trace:
        """Get aggregated maximum semblance as a Pyrocko trace.

        Returns:
            Trace: Holding the semblance
        """
        if padded:
            data = self.maximum_semblance
            start_time = self.start_time
        else:
            data = self.semblance_unpadded.max(axis=0)
            start_time = self.start_time - timedelta(
                seconds=int(round(self.padding_samples * self.sampling_rate))
            )

        return Trace(
            network="",
            station="semblance",
            tmin=start_time.timestamp(),
            deltat=1.0 / self.sampling_rate,
            ydata=data,
        )

    def reset(self) -> None:
        """Reset the semblance."""
        self.semblance_unpadded.fill(0.0)
        self._clear_cache()

    def _clear_cache(self) -> None:
        self._node_idx_max = None
        self._max_semblance = None

    def add(self, data: np.ndarray) -> None:
        """Add samblance matrix.

        Args:
            data (np.ndarray): Incoming semblance
        """
        self.semblance_unpadded += data
        self._clear_cache()

    def normalize(self, factor: int | float) -> None:
        """Normalize semblance by a factor.

        Args:
            factor (int | float): Normalization factor.
        """
        self.semblance_unpadded /= factor
        self._clear_cache()
