"""C extensions of qseek."""

from __future__ import annotations

from qseek.utils import check_simd_support

try:
    from qseek.ext._build import SIMD_FLAGS
except ImportError:  # Built without the generated build information
    SIMD_FLAGS: tuple[str, ...] = ()

# Warn before loading extensions compiled for SIMD features the CPU lacks
check_simd_support(SIMD_FLAGS)
