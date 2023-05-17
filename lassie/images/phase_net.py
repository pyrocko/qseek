from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import torch
from obspy import Stream
from pydantic import PositiveInt, PrivateAttr, conint
from pyrocko import obspy_compat
from seisbench.models import PhaseNet as PhaseNetSeisBench

from lassie.images.base import ImageFunction, WaveformImage
from lassie.utils import alog_call

obspy_compat.plant()

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


class PhaseNet(ImageFunction):
    image: Literal["PhaseNet"] = "PhaseNet"
    model: ModelName = "ethz"
    window_overlap_samples: conint(ge=1000, le=3000) = 2000
    torch_use_cuda: bool = False
    torch_cpu_threads: PositiveInt = 4
    seisbench_subprocesses: conint(ge=0) = 0
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
    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
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

        annotation_p = WaveformImage(
            image_function=self,
            phase=self.phase_map["P"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("P")],
        )
        annotation_s = WaveformImage(
            image_function=self,
            phase=self.phase_map["S"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("S")],
        )

        return [annotation_s, annotation_p]

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.phase_map.keys())
