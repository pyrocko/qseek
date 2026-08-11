import asyncio
import logging
from datetime import timedelta
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, PositiveFloat
from pyrocko.trace import Trace

from qseek.images.base import ImageFunction, PhaseName, WaveformImage
from qseek.utils import PhaseDescription

logger = logging.getLogger(__name__)


class StaLtaImage(WaveformImage): ...


class StaLta(ImageFunction):
    image: Literal["StaLta"] = "StaLta"

    sta_seconds: PositiveFloat = Field(
        default=0.5,
        description="Short-term average (STA) window length in seconds. "
        "Only used when `model` is `STA/LTA`.",
    )
    lta_seconds: PositiveFloat = Field(
        default=10.0,
        description="Long-term average (LTA) window length in seconds. "
        "Only used when `model` is `STA/LTA`.",
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

    async def prepare(self) -> None: ...

    async def process_traces(self, traces: list[Trace]) -> list[StaLtaImage]:
        """Process traces to generate image functions.

        Args:
            traces (list[Trace]): List of traces to process.

        Returns:
            list[WaveformImage]: List of image functions.
        """
        from obspy.signal.trigger import classic_sta_lta

        def _compute_characteristic_functions() -> list[Trace]:
            char_function_traces = []
            for tr in traces:
                obspy_tr = tr.to_obspy_trace()
                sampling_rate = obspy_tr.stats.sampling_rate
                sta_samples = max(1, int(self.stalta_sta_seconds * sampling_rate))
                lta_samples = max(
                    sta_samples + 1, int(self.stalta_lta_seconds * sampling_rate)
                )
                if obspy_tr.stats.npts <= lta_samples:
                    logger.warning(
                        "trace %s too short for STA/LTA (lta=%d samples, npts=%d)",
                        ".".join(tr.nslc_id),
                        lta_samples,
                        obspy_tr.stats.npts,
                    )
                    continue
                obspy_tr.data = classic_sta_lta(
                    obspy_tr.data.astype(np.float64),
                    sta_samples,
                    lta_samples,
                )
                char_function_traces.append(obspy_tr.to_pyrocko_trace())
            return char_function_traces

        char_function_traces = await asyncio.to_thread(
            _compute_characteristic_functions
        )

        p_traces = [tr for tr in char_function_traces if tr.channel.endswith("Z")]
        s_traces = [tr for tr in char_function_traces if not tr.channel.endswith("Z")]

        annotation_p = StaLtaImage(
            image_function=self.name,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            detection_half_width=self._detection_half_width(),
            traces=p_traces,
        )
        annotation_s = StaLtaImage(
            image_function=self.name,
            weight=self.weights["S"],
            phase=self.phase_map["S"],
            detection_half_width=self._detection_half_width(),
            traces=s_traces,
        )
        return [annotation_s, annotation_p]

    def get_blinding(self) -> timedelta:
        """Blinding duration for the image function. Added to padded waveforms.

        Returns:
            timedelta: The blinding duration for the image function.
        """
        raise NotImplementedError("must be implemented by subclass")

    def get_provided_phases(self) -> tuple[PhaseDescription, ...]:
        """Get the phases provided by the image function.

        Returns:
            tuple[PhaseDescription, ...]: The phases provided by the image function.
        """
        return ("P", "S")

        raise NotImplementedError("must be implemented by subclass")
