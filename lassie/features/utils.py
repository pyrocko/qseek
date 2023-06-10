from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrocko.trace import Trace


@dataclass
class ChannelSelector:
    channels: str
    number_channels: int

    def get_traces(self, traces: list[Trace]) -> list[Trace]:
        traces = [tr for tr in traces if tr.channel[-1] in self.channels]
        if len(traces) != self.number_channels:
            raise KeyError(
                "cannot get %d channels for selector %s",
                self.number_channels,
                self.channels,
            )
        return traces

    __call__ = get_traces


class ChannelSelectors:
    All = ChannelSelector("ENZ0123", 3)
    Horizontal = ChannelSelector("EN23", 2)
    Vertical = ChannelSelector("Z0", 1)
