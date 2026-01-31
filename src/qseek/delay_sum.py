from __future__ import annotations

from typing import NamedTuple

import numpy as np


class NodeStack(NamedTuple):
    """Delay-and-sum input of a single node.

    The extension functions in `qseek.ext.delay_sum` take a list of these.
    """

    shifts: np.ndarray
    """Shifts of the traces in samples, `np.int32` of shape `(n_traces,)`."""
    weights: np.ndarray
    """Weights of the traces, `np.float32` of shape `(n_traces,)`."""
    masked: bool = False
    """Masked nodes are skipped by the extension functions."""

    def mask(self, masked: bool = True) -> NodeStack:
        """Return a copy of the node with the mask set."""
        return self._replace(masked=masked)
