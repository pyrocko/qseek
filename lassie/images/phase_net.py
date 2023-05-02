from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from obspy import Stream
from pydantic import PrivateAttr, conint
from pyrocko import obspy_compat
from seisbench.models import PhaseNet as PhaseNetSeisBench

from lassie.images.base import ImageFunction, WaveformImage

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
    sampling_rate: float = 10.0
    overlap: conint(ge=100, le=3000) = 2000
    use_cuda: bool = False
    phase_map: dict[PhaseName, str] = {
        "P": "constant.P",
        "S": "constant.S",
    }

    _phase_net: PhaseNetSeisBench = PrivateAttr(None)

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self._phase_net = PhaseNetSeisBench.from_pretrained(self.model)
        if self.use_cuda:
            self._phase_net.cuda()
        self._phase_net.eval()

    def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        stream = Stream(tr.to_obspy_trace() for tr in traces)
        annotations: Stream = self._phase_net.annotate(stream)

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
        annotation_s.downsample(self.sampling_rate)
        annotation_p.downsample(self.sampling_rate)

        return [annotation_s, annotation_p]

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.phase_map.keys())
