from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import torch
from obspy import Stream
from pydantic import PositiveInt, PrivateAttr, conint
from pyrocko import obspy_compat
from seisbench import logger
from seisbench.models import PhaseNet as PhaseNetSeisBench

from lassie.images.base import ImageFunction, PickedArrival, WaveformImage
from lassie.utils import alog_call, to_datetime

obspy_compat.plant()

logger.setLevel(logging.CRITICAL)

if TYPE_CHECKING:
    from pyrocko.trace import Trace

ModelName = Literal[
    "diting",
    "ethz",
    "geofon",
    "instance",
    "iquique",
    "lendb",
    "neic",
    "obs",
    "original",
    "scedc",
    "stead",
]

PhaseName = Literal["P", "S"]
StackMethod = Literal["avg", "max"]


class PhaseNetPick(PickedArrival):
    provider: Literal["PhaseNetPick"] = "PhaseNetPick"
    phase: str
    detection_value: float


class PhaseNetImage(WaveformImage):
    def search_phase_arrival(
        self,
        trace_idx: int,
        modelled_arrival: datetime,
        search_length_seconds: float = 5,
        threshold: float = 0.1,
    ) -> PhaseNetPick | None:
        """Search for a peak in all station's image functions.

        Args:
            trace_idx (int): Index of the trace.
            modelled_arrival (datetime): Time to search around.
            search_length_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.

        Returns:
            datetime | None: Time of arrival, None is none found.
        """
        trace = self.traces[trace_idx]
        window_length = timedelta(seconds=search_length_seconds)
        search_trace = trace.chop(
            tmin=(modelled_arrival - window_length / 2).timestamp(),
            tmax=(modelled_arrival + window_length / 2).timestamp(),
            inplace=False,
        )
        time_seconds, value = search_trace.max()
        if value < threshold:
            return None
        return PhaseNetPick(
            time=to_datetime(time_seconds),
            detection_value=float(value),
            phase=self.phase,
        )


class PhaseNet(ImageFunction):
    image: Literal["PhaseNet"] = "PhaseNet"
    model: ModelName = "ethz"
    window_overlap_samples: conint(ge=1000, le=3000) = 2000
    torch_use_cuda: bool = False
    torch_cpu_threads: PositiveInt = 4
    stack_method: StackMethod = "avg"
    phase_map: dict[PhaseName, str] = {
        "P": "constant:P",
        "S": "constant:S",
    }

    _phase_net: PhaseNetSeisBench = PrivateAttr(None)

    def __init__(self, **data) -> None:
        super().__init__(**data)

        torch.set_num_threads(self.torch_cpu_threads)
        self._phase_net = PhaseNetSeisBench.from_pretrained(self.model)
        if self.torch_use_cuda:
            self._phase_net.cuda()
        self._phase_net.eval()

    @property
    def blinding(self) -> timedelta:
        blinding_samples = max(self._phase_net.default_args["blinding"])
        return timedelta(seconds=blinding_samples / 100)  # Hz PhaseNet sampling rate

    @alog_call
    async def process_traces(self, traces: list[Trace]) -> list[PhaseNetImage]:
        stream = Stream(tr.to_obspy_trace() for tr in traces)
        annotations: Stream = self._phase_net.annotate(
            stream,
            overlap=self.window_overlap_samples,
            # parallelism=self.seisbench_subprocesses,
        )

        annotated_traces: list[Trace] = [
            tr.to_pyrocko_trace()
            for tr in annotations
            if not tr.stats.channel.endswith("N")
        ]

        annotation_p = PhaseNetImage(
            image_function=self,
            phase=self.phase_map["P"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("P")],
        )
        annotation_s = PhaseNetImage(
            image_function=self,
            phase=self.phase_map["S"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("S")],
        )

        return [annotation_s, annotation_p]

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.phase_map.keys())
