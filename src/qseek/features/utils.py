from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrocko.trace import Trace


@dataclass
class ChannelSelector:
    channels: str
    number_channels: int
    normalize: bool = False

    def get_traces(self, traces: list[Trace]) -> list[Trace]:
        traces = [tr for tr in traces if tr.channel[-1] in self.channels]
        if len(traces) != self.number_channels:
            raise KeyError(
                f"cannot get {self.number_channels} channels"
                f" for selector {self.channels}"
                f" available: {', '.join('.'.join(tr.nslc_id) for tr in traces)}"
            )
        if self.normalize:
            traces_norm = traces[0].copy()
            traces_norm.ydata = np.linalg.norm(
                np.array([tr.ydata for tr in traces]), axis=0
            )
            return [traces_norm]
        return traces

    __call__ = get_traces


class ChannelSelectors:
    All = ChannelSelector("ENZ0123", 3)
    HorizontalAbs = ChannelSelector("EN23", 2, normalize=True)
    Vertical = ChannelSelector("Z0", 1)
    NorthEast = ChannelSelector("NE", 2)
