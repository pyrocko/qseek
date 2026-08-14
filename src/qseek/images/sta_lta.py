import asyncio
import logging
from datetime import datetime, timedelta
from typing import Annotated, Literal

import numpy as np
from obspy import Stream
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from pydantic import Field, PositiveFloat
from pyrocko.obspy_compat import to_obspy_stream, to_pyrocko_traces
from pyrocko.trace import NoData, Trace

from qseek.images.base import ImageFunction, ObservedArrival, PhaseName, WaveformImage
from qseek.utils import PhaseDescription, to_datetime

logger = logging.getLogger(__name__)


def _compute_characteristic_functions(
    stream: Stream,
    sta_seconds: float,
    lta_seconds: float,
) -> list[Trace]:
    char_function_traces = []
    for tr in stream:
        sampling_rate = tr.stats.sampling_rate
        sta_samples = max(1, int(sta_seconds * sampling_rate))
        lta_samples = max(sta_samples + 1, int(lta_seconds * sampling_rate))

        if tr.stats.npts <= lta_samples:
            logger.warning(
                "trace %s too short for STA/LTA (lta=%d samples, npts=%d)",
                ".".join(tr.nslc_id),
                lta_samples,
                tr.stats.npts,
            )
            continue
        tr.data = classic_sta_lta(
            tr.data.astype(np.float64),
            sta_samples,
            lta_samples,
        )
        char_function_traces.append(tr)

    return char_function_traces


class StaLtaImage(WaveformImage):
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
            logger.warning("No data to pick phase arrival %s.", ".".join(trace.nslc_id))
            return None

        trigger = trigger_onset(
            search_trace.data.astype(np.float64),
            threshold,
            threshold / 2,
        )
        if len(trigger) == 0:
            return None
        if len(trigger) > 1:
            logger.warning(
                "Multiple triggers found for %s, using the first one.",
                ".".join(trace.nslc_id),
            )
            return None
        trigger_on_idx = trigger[0][0]
        times = search_trace.get_xdata()
        trigger_time = times[trigger_on_idx]
        trigger_delay = trigger_time - event_time.timestamp()

        # Limit to post-event peaks
        post_event_peaks = trigger_delay > 0.0
        trigger_on_idx = trigger_on_idx[post_event_peaks]
        trigger_time = trigger_time[post_event_peaks]

        if not trigger_on_idx.size:
            return None

        return ObservedArrival(
            time=to_datetime(trigger_time[0]),
            detection_value=1.0,
            phase=self.phase,
        )


class StaLta(ImageFunction):
    """STA/LTA analytical characteristic function."""

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
        stream = to_obspy_stream(traces)

        char_function_traces = await asyncio.to_thread(
            _compute_characteristic_functions,
            stream,
            self.sta_seconds,
            self.lta_seconds,
        )

        traces = to_pyrocko_traces(char_function_traces)

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
