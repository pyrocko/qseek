from __future__ import annotations

from typing import Generic, Hashable, NamedTuple, TypeVar
from weakref import WeakSet

import numpy as np
from lru import LRU
from pydantic import BaseModel, ByteSize

_KT = TypeVar("_KT", bound=Hashable)
_T = TypeVar("_T", bound="ArrayLRUCache")

MB = 1024 * 1024
SIZE_MB = 1024 * MB

BARS = " ▂▃▄▅▆▇█"
NBARS = len(BARS)


class ArrayCaches(WeakSet[_T]):
    def get_fill_bytes(self) -> int:
        """Get the total size of all caches in bytes."""
        return sum(cache.size_bytes for cache in self)

    def get_fill_level(self) -> float:
        """Return the fill level of all caches as a float between 0.0 and 1.0."""
        return self.get_fill_bytes() / max(
            1, sum(cache._max_size_bytes for cache in self)
        )


CACHES: ArrayCaches[ArrayLRUCache] = ArrayCaches()


class CacheStats(NamedTuple):
    size: int
    hits: int
    misses: int
    hit_rate: float
    percent: float


class ArrayLRUCache(LRU, Generic[_KT]):
    size_bytes: int
    name: str
    short_name: str
    dtype: np.dtype

    _all_caches_bytes: int = 0

    def __init__(
        self,
        name: str,
        short_name: str = "",
        size_bytes: int = SIZE_MB,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(size=1, callback=self._remove_callback)
        self.size_bytes = 0
        self._max_size_bytes = size_bytes
        self.name = name
        self.short_name = short_name or "".join(w[0] for w in name.split("-")).upper()
        self.dtype = dtype

        CACHES.add(self)

    def set_name(self, name: str) -> None:
        """Set the name of the cache."""
        self.name = name

    def set_short_name(self, short_name: str) -> None:
        """Set the short name of the cache for display."""
        self.short_name = short_name

    def set_size_bytes(self, size_bytes: int) -> None:
        """Set the size of the cache in bytes."""
        self._max_size_bytes = size_bytes

    def get_size_bytes(self) -> int:
        """Get the size of the cache in bytes."""
        return self.size_bytes

    def __setitem__(self, key, value: np.ndarray) -> None:
        self.size_bytes += value.nbytes
        self._all_caches_bytes += value.nbytes
        value = value.astype(self.dtype)
        value.setflags(write=False)

        if self.size_bytes < self._max_size_bytes:
            self.set_size(len(self) + 1)
        super().__setitem__(key, value)

    def _remove_callback(self, key, value) -> None:
        self.size_bytes -= value.nbytes
        self._all_caches_bytes -= value.nbytes
        if self.size_bytes > self._max_size_bytes:
            self.set_size(len(self) - 1)

    def fill_level(self) -> float:
        """Return the fill level of the LUT as a float between 0.0 and 1.0."""
        return self.size_bytes / max(1, self._max_size_bytes)

    def hit_rate(self) -> float:
        hits, misses = super().get_stats()
        total_hits = hits + misses
        return hits / (total_hits or 1)

    def get_stats(self) -> CacheStats:
        hits, misses = super().get_stats()
        total_hits = hits + misses
        cache_hit_rate = hits / (total_hits or 1)
        return CacheStats(
            size=len(self),
            hits=hits,
            misses=misses,
            hit_rate=cache_hit_rate * 100,
            percent=(self.size_bytes / self._max_size_bytes) * 100,
        )

    def __rich__(self) -> str:
        bar_idx = self.fill_level() // (1 / NBARS)
        bar_idx = min(len(BARS) - 1, int(bar_idx))
        return f"{self.short_name}:{BARS[bar_idx]}"


class CachesStats(BaseModel):
    caches: dict[str, CacheStats]
    total_size: ByteSize

    @classmethod
    def get_stats(cls):
        caches = {cache.name: cache.get_stats() for cache in CACHES}
        total_size = sum(cache.size_bytes for cache in CACHES)
        return cls(caches=caches, total_size=ByteSize(total_size))


def get_stats() -> CachesStats:
    return CachesStats.get_stats()
