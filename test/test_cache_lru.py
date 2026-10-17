from __future__ import annotations

import numpy as np

from qseek.cache_lru import ArrayLRUCache


def stored_bytes(cache: ArrayLRUCache) -> int:
    return sum(value.nbytes for value in cache.values())


def test_size_bytes() -> None:
    entry = np.zeros(100, dtype=np.float64)
    entry_bytes = entry.astype(np.float32).nbytes
    cache = ArrayLRUCache(name="test", size_bytes=3 * entry_bytes)

    for key in range(10):
        cache[key] = entry
        assert cache.size_bytes == stored_bytes(cache)
    assert len(cache) == 3
    assert cache.size_bytes == 3 * entry_bytes
    assert cache.fill_level() == 1.0

    cache[9] = entry
    assert cache.size_bytes == 3 * entry_bytes

    del cache[9]
    assert cache.size_bytes == stored_bytes(cache) == 2 * entry_bytes

    assert cache.pop(8) is not None
    assert cache.pop(8, None) is None
    assert cache.size_bytes == stored_bytes(cache) == entry_bytes

    cache.clear()
    assert cache.size_bytes == 0


def test_hit_rate() -> None:
    cache = ArrayLRUCache(name="test")
    for key in range(5):
        cache[key] = np.zeros(10)
    assert cache.get_stats() == (0, 0)

    _ = cache[0]
    assert cache.hit_rate() == 1.0
