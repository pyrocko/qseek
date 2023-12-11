from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from obspy import Stream
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr
from pyrocko import obspy_compat
from seisbench import logger as seisbench_logger

from qseek.images.base import ImageFunction, PickedArrival, WaveformImage
from qseek.utils import alog_call, to_datetime

obspy_compat.plant()

seisbench_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from seisbench.models import PhaseNet as PhaseNetSeisBench

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
        search_length_seconds: float = 5.0,
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
    """PhaseNet image function. For more details see SeisBench documentation."""

    image: Literal["PhaseNet"] = "PhaseNet"
    model: ModelName = Field(
        default="ethz",
        description="SeisBench pre-trained PhaseNet model to use. "
        "Choose from `ethz`, `geofon`, `instance`, `iquique`, `lendb`, `neic`, `obs`,"
        " `original`, `scedc`, `stead`."
        " For more details see SeisBench documentation",
    )
    window_overlap_samples: int = Field(
        default=2000,
        ge=1000,
        le=3000,
        description="Window overlap in samples.",
    )
    torch_use_cuda: bool = Field(
        default=False,
        description="Use CUDA for inference.",
    )
    torch_cpu_threads: PositiveInt = Field(
        default=4,
        description="Number of CPU threads to use if only CPU is used.",
    )
    batch_size: int = Field(
        default=64,
        ge=64,
        description="Batch size for inference, larger values can improve performance.",
    )
    stack_method: StackMethod = Field(
        default="avg",
        description="Method to stack the overlaping blocks internally. "
        "Choose from `avg` and `max`.",
    )
    upscale_input: PositiveInt = Field(
        default=1,
        description="Upscale input by factor. "
        "This augments the input data from e.g. 100 Hz to 50 Hz (factor: `2`). Can be"
        " useful for high-frequency earthquake signals.",
    )
    phase_map: dict[PhaseName, str] = Field(
        default={
            "P": "constant:P",
            "S": "constant:S",
        },
        description="Phase mapping from SeisBench PhaseNet to Lassie phases.",
    )
    weights: dict[PhaseName, PositiveFloat] = Field(
        default={
            "P": 1.0,
            "S": 1.0,
        },
        description="Weights for each phase.",
    )

    _phase_net: PhaseNetSeisBench = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        import torch
        from seisbench.models import PhaseNet as PhaseNetSeisBench

        torch.set_num_threads(self.torch_cpu_threads)
        self._phase_net = PhaseNetSeisBench.from_pretrained(self.model)
        try:
            logger.info("compiling PhaseNet model...")
            self._phase_net = torch.compile(self._phase_net, mode="reduce-overhead")
        except RuntimeError as exc:
            logger.warning(
                "failed to compile PhaseNet model, using uncompiled model.",
                exc_info=exc,
            )
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
        if self.upscale_input > 1:
            scale = self.upscale_input
            for tr in stream:
                tr.stats.sampling_rate = tr.stats.sampling_rate / scale
        annotations: Stream = await asyncio.to_thread(
            self._phase_net.annotate,
            stream,
            overlap=self.window_overlap_samples,
            batch_size=self.batch_size,
        )

        if self.upscale_input > 1:
            scale = self.upscale_input
            for tr in annotations:
                tr.stats.sampling_rate = tr.stats.sampling_rate * scale

        annotated_traces: list[Trace] = [
            tr.to_pyrocko_trace()
            for tr in annotations
            if not tr.stats.channel.endswith("N")
        ]

        annotation_p = PhaseNetImage(
            image_function=self,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("P")],
        )
        annotation_s = PhaseNetImage(
            image_function=self,
            weight=self.weights["S"],
            phase=self.phase_map["S"],
            traces=[tr for tr in annotated_traces if tr.channel.endswith("S")],
        )

        return [annotation_s, annotation_p]

    def get_provided_phases(self) -> tuple[str, ...]:
        return tuple(self.phase_map.values())
