from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, ClassVar, Iterable

import numpy as np
from pydantic import computed_field
from pyrocko import parstack
from pyrocko.trace import Trace
from rich.table import Table
from scipy import signal

from qseek.ext.array_tools import apply_cache, fill_zero_bytes, fill_zero_bytes_mask
from qseek.stats import Stats
from qseek.utils import datetime_now, get_cpu_count, human_readable_bytes

if TYPE_CHECKING:
    from datetime import datetime

    from qseek.octree import Node


logger = logging.getLogger(__name__)


class SemblanceStats(Stats):
    total_nodes_stacked: int = 0
    total_stacking_time: timedelta = timedelta()
    last_nodes_stacked: int = 0
    last_stacking_time: timedelta = timedelta()
    semblance_size_bytes: int = 0

    _position: int = 30

    def add_stacking_time(self, calculation_time: timedelta, n_nodes: int) -> None:
        self.last_stacking_time = calculation_time
        self.last_nodes_stacked = n_nodes
        self.total_nodes_stacked += n_nodes
        self.total_stacking_time += calculation_time

    @computed_field
    @property
    def average_nodes_per_second(self) -> float:
        if not self.total_stacking_time:
            return 0.0
        return self.total_nodes_stacked / self.total_stacking_time.total_seconds()

    @computed_field
    @property
    def nodes_per_second(self) -> float:
        if not self.last_stacking_time:
            return 0.0
        return self.last_nodes_stacked / self.last_stacking_time.total_seconds()

    @computed_field
    @property
    def bytes_per_second(self) -> float:
        if not self.last_stacking_time:
            return 0.0
        return self.semblance_size_bytes / self.last_stacking_time.total_seconds()

    def _populate_table(self, table: Table) -> None:
        table.add_row(
            "Node stacking",
            f"{self.nodes_per_second:.0f} nodes/s"
            f" (avg {self.average_nodes_per_second:.1f} nodes/s)",
        )
        table.add_row(
            "Trace stacking",
            f"{human_readable_bytes(self.bytes_per_second)}/s",
        )
        table.add_row(
            "Semblance size",
            f"{human_readable_bytes(self.semblance_size_bytes)}"
            f" ({self.last_nodes_stacked} nodes)",
        )


class Semblance:
    _max_semblance: np.ndarray | None = None
    _node_max_idx: np.ndarray | None = None
    _node_hashes: list[bytes]
    _offset_samples: int = 0

    _stats: ClassVar[SemblanceStats] = SemblanceStats()
    _cached_semblance: ClassVar[np.ndarray | None] = None

    def __init__(
        self,
        nodes: Iterable[Node],
        n_samples: int,
        start_time: datetime,
        sampling_rate: float,
        padding_samples: int = 0,
        exponent: float = 1.0,
        cache: dict[bytes, np.ndarray] | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.padding_samples = padding_samples
        self.n_samples_unpadded = n_samples
        self.exponent = exponent

        self._start_time = start_time
        self._node_hashes = [node.hash() for node in nodes]
        n_nodes = len(self._node_hashes)

        n_values = n_nodes * n_samples

        if (
            self._cached_semblance is not None
            and self._cached_semblance.size >= n_values
        ):
            logger.debug("recycling semblance memory with paged NUMA memory")
            self.semblance_unpadded = self._cached_semblance[:n_values].reshape(
                (n_nodes, n_samples)
            )
            if cache:
                # If a cache is supplied only zero the missing nodes
                fill_zero_bytes_mask(
                    self.semblance_unpadded, ~self.get_cache_mask(cache)
                )
            else:
                fill_zero_bytes(self.semblance_unpadded)
        else:
            logger.info(
                "re-allocating semblance memory: %d", human_readable_bytes(n_values * 4)
            )
            self.semblance_unpadded = np.zeros((n_nodes, n_samples), dtype=np.float32)
            Semblance._cached_semblance = self.semblance_unpadded

        self._stats.semblance_size_bytes = self.semblance_unpadded.nbytes

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
            return self.semblance_unpadded[:, padding_samples:-padding_samples]
        return self.semblance_unpadded

    @property
    def start_time(self) -> datetime:
        """Start time of the volume."""
        return self._start_time + timedelta(
            seconds=self._offset_samples / self.sampling_rate
        )

    def get_cache(self) -> dict[bytes, np.ndarray]:
        """Return a cache dictionary containing the semblance data.

        We make a copy to keep the original data paged in memory.
        """
        cached_semblance = self.semblance_unpadded.copy()
        return {
            node_hash: cached_semblance[i, :]
            for i, node_hash in enumerate(self._node_hashes)
        }

    def get_semblance(self, time_idx: int) -> np.ndarray:
        """
        Get the semblance values at a specific time index.

        Parameters:
            time_idx (int): The index of the desired time.

        Returns:
            np.ndarray: The semblance values at the specified time index.
        """
        return self.semblance[:, time_idx]

    def get_cache_mask(self, cache: dict[bytes, np.ndarray]) -> np.ndarray:
        """
        Returns a boolean mask indicating whether each node hash
            in self._node_hashes is present in the cache.

        Args:
            cache (dict[bytes, np.ndarray]): The cache dictionary containing node
                hashes as keys.

        Returns:
            np.ndarray: A boolean mask indicating whether each node hash is
                present in the cache.
        """
        return np.array([hash in cache for hash in self._node_hashes])

    async def apply_cache(self, cache: dict[bytes, np.ndarray]) -> None:
        """
        Applies the cached data to the `semblance_unpadded` array.

        Args:
            cache (dict[bytes, np.ndarray]): The cache containing the cached data.

        Returns:
            None
        """
        if not cache:
            return
        mask = self.get_cache_mask(cache)
        logger.debug("applying cache for %d nodes", mask.sum())
        data = [cache[hash] for hash in self._node_hashes if hash in cache]

        # memoryview is faster then
        # self.semblance_unpadded[mask, :] = data
        # for idx, copy in enumerate(mask):
        #     if copy:
        #         memoryview(self.semblance_unpadded[idx])[:] = memoryview(data.pop(0))
        await asyncio.to_thread(
            apply_cache,
            self.semblance_unpadded,
            data,
            mask,
            nthreads=4,
        )

    def maximum_node_semblance(self) -> np.ndarray:
        semblance = self.semblance.max(axis=1)
        if self.exponent != 1.0:
            semblance **= self.exponent
        return semblance

    async def maxima_semblance(
        self,
        trim_padding: bool = True,
        nthreads: int = 12,
    ) -> np.ndarray:
        """Maximum semblance over time, aggregated over all nodes.

        Args:
            trim_padding (bool, optional): Trim padded data in post-processing.
            nparallel (int, optional): Number of threads for calculation. Defaults to 6.

        Returns:
            np.ndarray: Maximum semblance.
        """
        if self._max_semblance is None:
            node_idx = await self.maxima_node_idx(trim_padding=False, nthreads=nthreads)
            self._max_semblance = self.semblance_unpadded[
                node_idx, np.arange(self.n_samples_unpadded)
            ]
            self._max_semblance.setflags(write=False)

        if trim_padding:
            return self._max_semblance[self.padding_samples : -self.padding_samples]
        return self._max_semblance

    async def maxima_node_idx(
        self,
        trim_padding: bool = True,
        nthreads: int = 12,
    ) -> np.ndarray:
        """Indices of maximum semblance at any time step.

        Args:
            nparallel (int, optional): Number of threads for calculation. Defaults to 6.

        Returns:
            np.ndarray: Node indices.
        """
        if self._node_max_idx is None:
            self._node_max_idx = await asyncio.to_thread(
                parstack.argmax,
                self.semblance_unpadded,
                nparallel=nthreads,
            )
            self._node_max_idx.setflags(write=False)
        if trim_padding:
            return self._node_max_idx[self.padding_samples : -self.padding_samples]
        return self._node_max_idx

    def apply_exponent(self, exponent: float) -> None:
        """Apply exponent to the maximum semblance.

        Args:
            exponent (float): Exponent
        """
        if exponent == 1.0:
            return
        np.power(self.semblance_unpadded, exponent, out=self.semblance_unpadded)
        self._clear_cache()

    async def find_peaks(
        self,
        height: float,
        prominence: float,
        distance: float,
        trim_padding: bool = True,
        nthreads: int = 12,
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
        max_semblance_unpadded = await self.maxima_semblance(
            trim_padding=False, nthreads=nthreads
        )

        detection_idx, _ = await asyncio.to_thread(
            signal.find_peaks,
            max_semblance_unpadded,
            height=height,
            prominence=prominence,
            distance=distance,
        )
        if trim_padding:
            max_semblance_trimmed = await self.maxima_semblance()

            detection_idx -= self.padding_samples
            detection_idx = detection_idx[detection_idx >= 0]
            detection_idx = detection_idx[detection_idx < max_semblance_trimmed.size]
            semblance = max_semblance_trimmed[detection_idx]
        else:
            semblance = max_semblance_unpadded[detection_idx]

        return detection_idx, semblance

    async def get_trace(self, padded: bool = True) -> Trace:
        """Get aggregated maximum semblance as a Pyrocko trace.

        Returns:
            Trace: Holding the semblance
        """
        if padded:
            data = await self.maxima_semblance(trim_padding=True)
            start_time = self.start_time
        else:
            data = await self.maxima_semblance(trim_padding=False)
            start_time = self.start_time - timedelta(
                seconds=int(round(self.padding_samples * self.sampling_rate))
            )

        return Trace(
            network="",
            station="semblance",
            tmin=start_time.timestamp(),
            deltat=1.0 / self.sampling_rate,
            ydata=data.astype(np.float32, copy=False),
        )

    def _clear_cache(self) -> None:
        self._node_max_idx = None
        self._max_semblance = None

    async def calculate_semblance(
        self,
        trace_data: list[np.ndarray],
        offsets: np.ndarray,
        shifts: np.ndarray,
        weights: np.ndarray,
        threads: int = 0,
    ) -> None:
        # Hold threads back for I/O
        threads = threads or max(1, get_cpu_count() - 6)

        start_time = datetime_now()
        _, offset_samples = await asyncio.to_thread(
            parstack.parstack,
            arrays=trace_data,
            offsets=offsets,
            shifts=shifts,
            weights=weights,
            lengthout=self.n_samples_unpadded,
            result=self.semblance_unpadded,
            dtype=self.semblance_unpadded.dtype,
            method=0,
            nparallel=threads,
        )
        self._stats.add_stacking_time(datetime_now() - start_time, self.n_nodes)
        if self._offset_samples and self._offset_samples != offset_samples:
            logging.warning(
                "offset samples changed from %d to %d",
                self._offset_samples,
                offset_samples,
            )
        self._offset_samples = offset_samples
        self._clear_cache()

    def normalize(
        self,
        factor: int | float,
        semblance_cache: dict[bytes, np.ndarray] | None = None,
    ) -> None:
        """Normalize semblance by a factor.

        Args:
            factor (int | float): Normalization factor.
        """
        if factor == 1.0:
            return
        if semblance_cache:
            cache_mask = self.get_cache_mask(semblance_cache)
            self.semblance_unpadded[~cache_mask] /= factor
        else:
            self.semblance_unpadded /= factor
        self._clear_cache()
