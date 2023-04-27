import logging
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from pyrocko import trace

from lassie.ifcs import IFC
from lassie.octree import Octree

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from lassie.models.receiver import Receiver

logger = logging.getLogger(__name__)


def zero_fill_traces(traces: list[Trace], tmin: float, tmax: float) -> list[Trace]:
    traces_out = []
    traces = trace.degapper(traces)

    trs_groups = defaultdict(list)
    for tr in traces:
        trs_groups[tr.nslc_id].append(tr)

    for trs_group in trs_groups.values():
        deltats, dtypes = zip((tr.deltat, tr.ydata.dtype) for tr in trs_group)

        if len(set(deltats)) > 1:
            logger.warn("inconsistent sample rate, cannot merge traces")
            continue
        if len(set(dtypes)) > 1:
            logger.warn("inconsistent data type, cannot merge traces")
            continue

        tr_combined = trs_group[0].copy()
        tr_combined.extend(tmin, tmax, fillmethod="zeros")
        for tr in trs_group[1:]:
            tr_combined.add(tr)
        traces_out.append(tr_combined)

    return traces_out


class SearchBlock:
    def __init__(
        self,
        traces: list[Trace],
        receivers: list[Receiver],
        ifcs: list[IFC],
        octree: Octree,
        padding: float,
    ) -> None:
        self.traces = traces
        self.octree = octree
        self.ifcs = ifcs

        self.clean_traces()

    @cached_property
    def tmin(self) -> float:
        return self.get_time_span()[0]

    @cached_property
    def tmax(self) -> float:
        return self.get_time_span()[1]

    def get_time_span(self) -> tuple[float, float]:
        tmins, tmaxs = zip((tr.tmin, tr.tmax) for tr in self.traces)
        return min(tmins), max(tmaxs)

    def clean_traces(self) -> None:
        for tr in self.traces.copy():
            if tr.ydata.size == 0 or not np.all(np.isfinite(tr.ydata)):
                logger.warn("skipping empty trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)

        self.traces = zero_fill_traces(self.traces, self.tmin, self.tmax)

    def calculate_cfs(self) -> list[Trace]:
        for ifc in self.ifcs:
            ifc.preprocess(self.traces, self.tmin, self.tmax)

    def search(self):
        ...
