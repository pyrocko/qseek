from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np
from obspy import Stream
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr
from pyrocko import obspy_compat
from pyrocko.trace import NoData
from scipy import signal
from seisbench import logger as seisbench_logger

from qseek.images.base import ImageFunction, ObservedArrival, WaveformImage
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


class PhaseNetImage(WaveformImage):
    def search_phase_arrival(
        self,
        trace_idx: int,
        event_time: datetime,
        modelled_arrival: datetime,
        search_window_seconds: float = 5.0,
        threshold: float = 0.1,
        detection_blinding_seconds: float = 1.0,
    ) -> ObservedArrival | None:
        """Search for the closest peak (pick) in the station's image functions.

        Args:
            trace_idx (int): Index of the trace.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.
            search_window_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.
            detection_blinding_seconds (float, optional): Blinding time in seconds for
                the peak detection. Defaults to 1 second.

        Returns:
            datetime | None: Time of arrival, None is none found.
        """
        trace = self.traces[trace_idx]
        window_length = timedelta(seconds=search_window_seconds)
        try:
            search_trace = trace.chop(
                tmin=(modelled_arrival - window_length / 2).timestamp(),
                tmax=(modelled_arrival + window_length / 2).timestamp(),
                inplace=False,
            )
        except NoData:
            logger.warning("No data to pick phase arrival.")
            return None

        peak_idx, _ = signal.find_peaks(
            search_trace.ydata,
            height=threshold,
            prominence=threshold,
            distance=int(detection_blinding_seconds * 1 / search_trace.deltat),
        )
        if False:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            time = search_trace.get_xdata()
            std = np.std(search_trace.get_ydata())

            ax.plot(time, search_trace.get_ydata())
            ax.grid(alpha=0.3)
            ax.axhline(threshold, color="r", linestyle="--", label="threshold")
            ax.axhline(std, color="g", linestyle="--", label="std")
            ax.axhline(3 * std, color="b", linestyle="dotted", label="3*std")
            ax.axvline(
                modelled_arrival.timestamp(),
                color="k",
                alpha=0.3,
                label="modelled arrival",
            )
            if peak_idx.size:
                ax.axvline(time[peak_idx], color="m", linestyle="--", label="peaks")
            plt.show()

        times = search_trace.get_xdata()
        peak_times = times[peak_idx]
        peak_delay = peak_times - event_time.timestamp()

        # Limit to post-event peaks
        after_event_peaks = peak_delay > 0.0
        peak_idx = peak_idx[after_event_peaks]
        peak_times = peak_times[after_event_peaks]
        peak_delay = peak_delay[after_event_peaks]

        if not peak_idx.size:
            return None

        peak_values = search_trace.get_ydata()[peak_idx]
        closest_peak_idx = np.argmin(peak_delay)

        return ObservedArrival(
            time=to_datetime(peak_times[closest_peak_idx]),
            detection_value=peak_values[closest_peak_idx],
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
        default=1500,
        ge=1000,
        le=3000,
        description="Window overlap in samples.",
    )
    torch_use_cuda: bool | int = Field(
        default=False,
        description="Use CUDA for inference. If `True` use default device, if `int` use"
        " the specified device.",
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
    rescale_input: PositiveFloat = Field(
        default=1.0,
        description="Upscale input by factor. "
        "This augments the input data from e.g. 100 Hz to 50 Hz (factor: `2`). Can be"
        " useful for high-frequency microseismic events.",
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

    @property
    def phase_net(self) -> PhaseNetSeisBench:
        if self._phase_net is None:
            self._prepare()
        return self._phase_net

    def _prepare(self) -> None:
        import torch
        from seisbench.models import PhaseNet as PhaseNetSeisBench

        torch.set_num_threads(self.torch_cpu_threads)
        self._phase_net = PhaseNetSeisBench.from_pretrained(self.model)
        if self.torch_use_cuda:
            if isinstance(self.torch_use_cuda, bool):
                self._phase_net.cuda()
            else:
                self._phase_net.cuda(self.torch_use_cuda)
        self._phase_net.eval()
        try:
            logger.info("compiling PhaseNet model...")
            self._phase_net = torch.compile(self._phase_net, mode="max-autotune")
        except RuntimeError as exc:
            logger.warning(
                "failed to compile PhaseNet model, using uncompiled model.",
                exc_info=exc,
            )

    def get_blinding(self, sampling_rate: float) -> timedelta:
        blinding_samples = (
            max(self.phase_net.default_args["blinding"]) / self.rescale_input
        )
        return timedelta(seconds=blinding_samples / sampling_rate)

    @alog_call
    async def process_traces(self, traces: list[Trace]) -> list[PhaseNetImage]:
        stream = Stream(tr.to_obspy_trace() for tr in traces)
        if self.rescale_input > 1:
            scale = self.rescale_input
            for tr in stream:
                tr.stats.sampling_rate = tr.stats.sampling_rate / scale

        annotations: Stream = await asyncio.to_thread(
            self.phase_net.annotate,
            stream,
            overlap=self.window_overlap_samples,
            batch_size=self.batch_size,
            copy=False,
        )

        if self.rescale_input > 1:
            scale = self.rescale_input
            for tr in annotations:
                tr.stats.sampling_rate = tr.stats.sampling_rate * scale
                blinding_samples = self.phase_net.default_args["blinding"][0]
                # 100 Hz is the native sampling rate of PhaseNet
                blinding_seconds = (blinding_samples / 100.0) * (1.0 - 1 / scale)
                tr.stats.starttime -= blinding_seconds

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
