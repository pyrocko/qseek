from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchSizeTuner:
    """Climb to the batch size with the highest throughput, in factors of two.

    A probe alternates the calls between the current batch size and its double,
    or its half, so that both see similar data and load. The probe becomes the
    batch size when its median throughput is higher by more than `min_gain`, and
    the climb continues in that direction. When neither neighbor is faster, the
    batch size is kept for `probe_interval` calls. The first call with a batch
    size is a warm-up, e.g. for compilation, and is not measured.
    """

    batch_size: int = 256
    min_batch_size: int = 32
    max_batch_size: int = 4096
    n_measurements: int = 3
    min_gain: float = 0.05
    probe_interval: int = 100

    _warm: set[int] = field(default_factory=set)
    _rates: dict[int, list[float]] = field(default_factory=dict)
    _probe: int | None = None
    _turned: bool = False
    _wait: int = 0

    def next_batch_size(self) -> int:
        """Batch size for the next call."""
        probe = self._probe
        if probe and len(self._rates[probe]) <= len(self._rates[self.batch_size]):
            return probe
        return self.batch_size

    def record(self, batch_size: int, n_windows: int, seconds: float) -> None:
        """Record the number of windows a call processed and its duration."""
        if n_windows <= 0 or seconds <= 0.0:
            return
        if batch_size not in self._warm:
            self._warm.add(batch_size)
            return
        if self._probe is None:
            self._wait -= 1
            if self._wait <= 0:
                self._climb(up=True, turned=False)
            return
        if batch_size in self._rates:
            self._rates[batch_size].append(n_windows / seconds)

        probe, current = self._probe, self.batch_size
        if (
            min(len(self._rates[probe]), len(self._rates[current]))
            < self.n_measurements
        ):
            return
        gain = median(self._rates[probe]) / median(self._rates[current]) - 1.0
        logger.info(
            "batch size %d vs %d: %+.1f%% throughput", probe, current, gain * 100
        )
        if gain > self.min_gain:
            self.batch_size = probe
            self._climb(up=probe > current, turned=True)
        elif self._turned:
            self._settle()
        else:
            self._climb(up=probe < current, turned=True)

    def record_out_of_memory(self, batch_size: int) -> bool:
        """Exclude the batch size and larger ones; False if none smaller is left."""
        if batch_size <= self.min_batch_size:
            return False
        self.max_batch_size = min(self.max_batch_size, batch_size // 2)
        self.batch_size = min(self.batch_size, self.max_batch_size)
        logger.warning(
            "out of memory, limiting the batch size to %d", self.max_batch_size
        )
        self._settle()
        return True

    def _climb(self, up: bool, turned: bool) -> None:
        """Probe the neighbor; at the limits turn around once, then settle."""
        size = self.batch_size * 2 if up else self.batch_size // 2
        if self.min_batch_size <= size <= self.max_batch_size:
            self._probe, self._turned = size, turned
            self._rates = {self.batch_size: [], size: []}
        elif not turned:
            self._climb(not up, turned=True)
        else:
            self._settle()

    def _settle(self) -> None:
        logger.info("using batch size %d", self.batch_size)
        self._probe = None
        self._wait = self.probe_interval
