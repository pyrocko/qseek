from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from obspy import Stream
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
)
from pyrocko import obspy_compat
from pyrocko.trace import NoData
from scipy import signal
from seisbench import logger as seisbench_logger

from qseek.images.base import (
    ImageFunction,
    ObservedArrival,
    PhaseName,
    Picker,
    WaveformImage,
)
from qseek.types import FilePath
from qseek.utils import PhaseDescription, alog_call, to_datetime

obspy_compat.plant()

seisbench_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from seisbench.models import WaveformModel


ModelName = Literal[
    "PhaseNet",
    "EQTransformer",
    "OBSTransformer",
    "LFEDetect",
    "GPD",
]


PreTrainedName = Literal[
    "cascadia",
    "cms",
    "diting",
    "dummy",
    "ethz",
    "geofon",
    "instance",
    "iquique",
    "jcms",
    "jcs",
    "jms",
    "lendb",
    "mexico",
    "nankai",
    "neic",
    "obs",
    "obst2024",
    "original",
    "original_nonconservative",
    "san_andreas",
    "scedc",
    "stead",
    "volpick",
]

StackMethod = Literal["avg", "max"]


class AnnotationPicker(Picker):
    """Pick phase arrivals from SeisBench annotations.

    The pick is the annotation peak closest to the modeled arrival time within the
    search window. Peaks before the event origin time are rejected.
    """

    threshold_p: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Minimum height and prominence of a P phase annotation peak.",
    )
    threshold_s: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Minimum height and prominence of an S phase annotation peak.",
    )
    search_window_seconds: PositiveFloat = Field(
        default=5.0,
        description="Total length of the search window in seconds, centered on the"
        " modeled arrival time.",
    )
    peak_separation_seconds: NonNegativeFloat = Field(
        default=0.1,
        description="Minimum separation between annotation peaks in seconds.",
    )

    def get_threshold(self, phase: PhaseName) -> float:
        """Get the peak threshold for a SeisBench phase.

        Args:
            phase (PhaseName): SeisBench annotation phase, `P` or `S`.

        Returns:
            float: Peak threshold.
        """
        match phase:
            case "P":
                return self.threshold_p
            case "S":
                return self.threshold_s
            case _:
                raise ValueError(f"No pick threshold for phase `{phase}`.")

    def pick_trace(
        self,
        trace: Trace,
        phase: PhaseDescription,
        event_time: datetime,
        modelled_arrival: datetime,
    ) -> ObservedArrival | None:
        """Pick the annotation peak closest to the modelled arrival.

        Args:
            trace (Trace): Annotation trace, its channel is the SeisBench phase.
            phase (PhaseDescription): Phase of the observed arrival.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.

        Returns:
            ObservedArrival | None: Picked arrival, None if none found.
        """
        threshold = self.get_threshold(trace.channel)
        half_window = timedelta(seconds=self.search_window_seconds / 2)
        try:
            search_trace = trace.chop(
                tmin=(modelled_arrival - half_window).timestamp(),
                tmax=(modelled_arrival + half_window).timestamp(),
                inplace=False,
            )
        except NoData:
            logger.warning("No data to pick phase arrival %s.", ".".join(trace.nslc_id))
            return None

        peak_idx, _ = signal.find_peaks(
            search_trace.ydata,
            height=threshold,
            prominence=threshold,
            distance=max(1, self.peak_separation_seconds / search_trace.deltat),
        )
        peak_times = search_trace.get_xdata()[peak_idx]

        # Limit to post-event peaks
        post_event_peaks = peak_times > event_time.timestamp()
        peak_idx = peak_idx[post_event_peaks]
        peak_times = peak_times[post_event_peaks]
        if not peak_idx.size:
            return None

        closest_peak = np.argmin(np.abs(peak_times - modelled_arrival.timestamp()))
        return ObservedArrival(
            time=to_datetime(peak_times[closest_peak]),
            detection_value=float(search_trace.ydata[peak_idx[closest_peak]]),
            phase=phase,
        )


class SeisBench(ImageFunction):
    """Phase annotations from machine learning pickers in SeisBench.

    The image is the probability of a P or S phase arrival, as annotated by a
    pre-trained SeisBench model, e.g. PhaseNet or EQTransformer.
    """

    image: Literal["SeisBench"] = "SeisBench"

    model: ModelName = Field(
        default="PhaseNet",
        description="The SeisBench model.",
    )

    pretrained: PreTrainedName | FilePath = Field(
        default="original",
        description=(
            'The pre-trained weights of the model, e.g. `"original"`, `"ethz"`, '
            '`"instance"` or `"stead"`, or the path to a custom model `.json` file. The'
            " [SeisBench documentation](https://seisbench.readthedocs.io/) lists which "
            "weights are available for which model."
        ),
    )
    window_overlap_samples: int = Field(
        default=2000,
        ge=1000,
        le=3000,
        description="Window overlap in samples.",
    )
    torch_use_cuda: bool | int = Field(
        default=True,
        description="Use CUDA for the inference. `true` uses the default device, a"
        " number selects the device, e.g. `0` for the first one. `false` runs on the"
        " CPU.",
    )
    torch_cpu_threads: PositiveInt = Field(
        default=4,
        description="Number of CPU threads to use if only CPU is used.",
    )
    batch_size: int = Field(
        default=128,
        ge=64,
        description="Batch size for inference, larger values can improve performance.",
    )
    stack_method: StackMethod = Field(
        default="avg",
        description=(
            "How overlapping annotation windows are combined, by their average "
            '(`"avg"`) or maximum (`"max"`).'
        ),
    )
    sampling_rate: PositiveFloat | Literal["input"] = Field(
        default=100.0,
        description=(
            "Sampling rate in Hz that the model assumes for its input. A rate above the"
            " native rate of the model, e.g. 200 Hz for a model trained at 100 Hz, "
            "rescales the input by their ratio. This can help to detect high-frequency "
            'microseismic events. `"input"` uses the sampling rate of the input '
            "traces, which must all have the same rate. Until the first traces are "
            "processed, the blinding assumes the native rate of the model."
        ),
    )
    phase_map: dict[PhaseName, str] = Field(
        default={
            "P": "cake:P",
            "S": "cake:S",
        },
        description="Phase mapping from SeisBench PhaseNet to "
        "Qseek travel time phases.",
    )
    weights: dict[PhaseName, Annotated[float, Field(strict=True, ge=0.0)]] = Field(
        default={
            "P": 1.0,
            "S": 1.0,
        },
        description="Weights for each phase.",
    )
    picker: AnnotationPicker = Field(
        default_factory=AnnotationPicker,
        description="Picker to use for the image function.",
    )

    _seisbench_model: WaveformModel = PrivateAttr()
    _native_sampling_rate: PositiveFloat = PrivateAttr(100.0)
    _rescale_input: PositiveFloat = PrivateAttr(1.0)
    _padded_blinding: timedelta | None = PrivateAttr(None)

    @property
    def seisbench_model(self) -> WaveformModel:
        return self._seisbench_model

    async def prepare(self) -> None:
        logger.info("preparing SeisBench image function...")

        import seisbench.models as sbm
        import torch

        torch.set_num_threads(self.torch_cpu_threads)

        match self.model:
            case "PhaseNet":
                model = sbm.PhaseNet
            case "EQTransformer":
                model = sbm.EQTransformer
            case "GPD":
                model = sbm.GPD
            case "OBSTransformer":
                model = sbm.OBSTransformer
            case "LFEDetect":
                model = sbm.LFEDetect
            case _:
                raise ValueError(f"Model `{self.model}` not available.")

        if isinstance(self.pretrained, Path):
            # SeisBench uses an incomplete filename for loading from file
            logger.info("loading local SeisBench model from %s", self.pretrained)
            self._seisbench_model = model.load(self.pretrained.with_suffix(""))
        else:
            logger.info("loading pre-trained SeisBench model %s...", self.pretrained)
            self._seisbench_model = model.from_pretrained(self.pretrained, update=False)
        self._native_sampling_rate = self._seisbench_model.sampling_rate
        if self.sampling_rate != "input":
            self._set_model_sampling_rate(self.sampling_rate)
        # 0 selects the first device, only False runs on the CPU
        if self.torch_use_cuda is not False:
            try:
                if isinstance(self.torch_use_cuda, bool):
                    self._seisbench_model.cuda()
                else:
                    self._seisbench_model.cuda(self.torch_use_cuda)
                logger.info("using CUDA for SeisBench model")
            except (RuntimeError, AssertionError) as exc:
                logger.warning(
                    "failed to use CUDA for SeisBench model, using CPU",
                    exc_info=exc,
                )

        self._seisbench_model.eval()
        try:
            logger.info("compiling SeisBench model...")
            self._seisbench_model = torch.compile(
                self._seisbench_model,
                mode="max-autotune",
            )
        except RuntimeError as exc:
            logger.warning(
                "failed to compile SeisBench model, using uncompiled model.",
                exc_info=exc,
            )

    def _set_model_sampling_rate(self, sampling_rate: float) -> None:
        """Set the sampling rate the model assumes and the input rescaling."""
        # torch.compile wraps the model, the attribute has to be set on the original
        model = getattr(self._seisbench_model, "_orig_mod", self._seisbench_model)
        model.sampling_rate = sampling_rate
        self._rescale_input = sampling_rate / self._native_sampling_rate
        logger.debug("rescaling SeisBench input by factor %.2f", self._rescale_input)

    def get_blinding_samples(self) -> tuple[int, int]:
        if self.model == "GPD":
            return (0, 0)
        try:
            return self.seisbench_model.default_args["blinding"]
        except KeyError:
            return self.seisbench_model._annotate_args["blinding"][1]

    def get_blinding(self) -> timedelta:
        sampling_rate = (
            self._seisbench_model.sampling_rate
            if self.sampling_rate == "input"
            else self.sampling_rate
        )
        blinding = timedelta(seconds=max(self.get_blinding_samples()) / sampling_rate)
        if self._padded_blinding is None:
            # The search pads the waveforms with the first value, before any data
            self._padded_blinding = blinding
        return blinding

    def _detection_half_width(self) -> float:
        """Half width of the detection window in seconds."""
        # The 0.2 seconds is the default value from SeisBench training
        return 0.2 / self._rescale_input

    @alog_call
    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        if self.sampling_rate == "input" and traces:
            rates = {round(1.0 / tr.deltat, 6) for tr in traces}
            if len(rates) != 1:
                raise ValueError(
                    "SeisBench `sampling_rate='input'` requires traces with a "
                    f"homogeneous sampling rate, got {sorted(rates)} Hz."
                )
            self._set_model_sampling_rate(rates.pop())
            blinding = self.get_blinding()
            if self._padded_blinding and blinding > self._padded_blinding:
                logger.warning(
                    "input sampling rate yields a blinding of %s, longer than the "
                    "%s the window padding was computed with. Annotations at the "
                    "window edges can be affected, set `sampling_rate` explicitly.",
                    blinding,
                    self._padded_blinding,
                )
                self._padded_blinding = blinding

        stream = Stream(tr.to_obspy_trace() for tr in traces)

        annotations: Stream = await asyncio.to_thread(
            self.seisbench_model.annotate,
            stream,
            overlap=self.window_overlap_samples,
            batch_size=self.batch_size,
            stacking=self.stack_method,
            copy=False,
        )

        annotated_traces: list[Trace] = [
            tr.to_pyrocko_trace()
            for tr in annotations
            if tr.stats.channel.endswith("P") or tr.stats.channel.endswith("S")
        ]

        annotation_p = WaveformImage(
            image_function=self.name,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            detection_half_width=self._detection_half_width(),
            traces=[tr for tr in annotated_traces if tr.channel.endswith("P")],
        )
        annotation_s = WaveformImage(
            image_function=self.name,
            weight=self.weights["S"],
            phase=self.phase_map["S"],
            detection_half_width=self._detection_half_width(),
            traces=[tr for tr in annotated_traces if tr.channel.endswith("S")],
        )

        for tr in annotation_s.traces + annotation_p.traces:
            tr.set_channel(tr.channel[-1])

        return [annotation_s, annotation_p]

    def get_phases(self) -> tuple[str, ...]:
        return tuple(self.phase_map.values())
