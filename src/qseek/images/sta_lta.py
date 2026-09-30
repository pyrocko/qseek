import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal

import numpy as np
from obspy import Stream
from obspy.signal.trigger import trigger_onset
from pydantic import Field, PositiveFloat
from pyrocko.obspy_compat import to_pyrocko_traces
from pyrocko.trace import NoData, Trace
from scipy.signal import hilbert
from scipy.stats import median_abs_deviation

from qseek.images.base import ImageFunction, ObservedArrival, PhaseName, WaveformImage
from qseek.utils import PhaseDescription, to_datetime

# The signal transformation, the STA/LTA computation, and the merging of multiple horizontal components
# are based on the implementation in QuakeMigrate[1]: https://github.com/QuakeMigrate/QuakeMigrate
# [1] Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular,
# Open-Source Python Package for Automatic Earthquake Detection and Location. In AGU Fall Meeting 2020. AGU.

logger = logging.getLogger(__name__)

SignalTransform = Literal["energy", "absolute", "envelope"]


def _transform_signal(data: np.ndarray, method: SignalTransform) -> np.ndarray:
    """Transform a waveform into a non-negative characteristic signal.

    Args:
        data (np.ndarray): Raw waveform amplitude.
        method (SignalTransform): `energy` (squared amplitude), `absolute`
            (absolute amplitude), or `envelope` (amplitude of the analytic
            signal, i.e. the Hilbert envelope).

    Returns:
        np.ndarray: Non-negative transformed signal.
    """
    if method == "energy":
        return data**2
    if method == "absolute":
        return np.abs(data)
    return np.abs(hilbert(data))


def _overlapping_sta_lta(signal: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
    """Classic (right-aligned) STA/LTA ratio of an already non-negative signal.

    Args:
        signal (np.ndarray): Non-negative characteristic signal.
        nsta (int): Number of samples in the short-term window.
        nlta (int): Number of samples in the long-term window.

    Returns:
        np.ndarray: STA/LTA ratio, computed in overlapping windows.
    """
    sta = np.cumsum(signal, dtype=np.float64)
    lta = sta.copy()

    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad with ones (= null result) where the LTA window is not yet full.
    sta[: nlta - 1] = 1.0
    lta[: nlta - 1] = 1.0

    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = dtiny

    return sta / lta


def _merge_horizontal_components(traces: list[Trace]) -> list[Trace]:
    """Combine multiple horizontal-component onset traces per station.

    Components sharing the same network/station/location are combined as the
    root-mean-square of their STA/LTA onset functions.

    Args:
        traces (list[Trace]): Per-component STA/LTA onset traces.

    Returns:
        list[Trace]: One onset trace per station.
    """
    grouped: dict[tuple[str, str, str], list[Trace]] = {}
    for tr in traces:
        grouped.setdefault((tr.network, tr.station, tr.location), []).append(tr)

    merged_traces = []
    for group in grouped.values():
        if len(group) == 1:
            merged_traces.append(group[0])
            continue
        stacked = np.array([tr.ydata for tr in group])
        rms = np.sqrt(np.sum(stacked**2, axis=0) / len(group))
        merged = group[0].copy()
        merged.set_ydata(rms)
        merged_traces.append(merged)

    return merged_traces


def _compute_characteristic_functions(
    stream: Stream,
    sta_seconds: float,
    lta_seconds: float,
    signal_transform: SignalTransform,
) -> list[Trace]:
    """Compute the STA/LTA characteristic function.

    For each trace, the waveform is first transformed into a non-negative
    characteristic signal (`signal_transform`), then the short-term and
    long-term average windows are converted from seconds to samples based on
    the sampling rate, and the classic STA/LTA ratio is computed.
    Traces that are shorter than the required LTA window are skipped and a warning is logged.

    Args:
        stream (Stream): containing the input seismic traces
        sta_seconds (float): Duration of the short-term average window in seconds.
        lta_seconds (float): Duration of the long-term average window in seconds.
        signal_transform (SignalTransform): Transform applied to the waveform
            before computing the STA/LTA ratio.

    Returns:
        list of Traces: A list of traces containing the STA/LTA characteristic
        functions.
    """
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
        transformed = _transform_signal(tr.data.astype(np.float64), signal_transform)
        tr.data = _overlapping_sta_lta(transformed, sta_samples, lta_samples)
        char_function_traces.append(tr)

    return char_function_traces


@dataclass
class StaLtaImage(WaveformImage):
    mad_factor: float = 10.0

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

        The trigger threshold is derived from the median absolute deviation
        (MAD) of the station's full STA/LTA trace, scaled by `mad_factor`.

        Args:
            trace_idx (int): Index of the trace.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.
            search_window_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Unused, kept for interface compatibility
                with other image functions. The MAD-based threshold is used instead.
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

        mad_threshold = median_abs_deviation(trace.ydata) * self.mad_factor

        trigger = trigger_onset(
            search_trace.ydata.astype(np.float64),
            mad_threshold,
            mad_threshold / 2,
        )
        if len(trigger) == 0:
            return None
        if len(trigger) > 1:
            logger.debug(
                "%d triggers found for %s, picking the one closest to the "
                "modelled arrival.",
                len(trigger),
                ".".join(trace.nslc_id),
            )
        trigger_on_idx = trigger[:, 0]
        times = search_trace.get_xdata()
        trigger_times = times[trigger_on_idx]
        trigger_delays = trigger_times - event_time.timestamp()

        # Limit to post-event peaks
        post_event_peaks = trigger_delays > 0.0
        trigger_on_idx = trigger_on_idx[post_event_peaks]
        trigger_times = trigger_times[post_event_peaks]

        if not trigger_on_idx.size:
            return None

        detection_values = search_trace.ydata[trigger_on_idx]

        # Pick the trigger onset closest to the modelled arrival
        residuals = trigger_times - modelled_arrival.timestamp()
        closest_idx = np.argmin(np.abs(residuals))

        return ObservedArrival(
            time=to_datetime(trigger_times[closest_idx]),
            detection_value=float(detection_values[closest_idx]),
            phase=self.phase,
        )


class StaLta(ImageFunction):
    """STA/LTA analytical characteristic function."""

    image: Literal["StaLta"] = "StaLta"

    sta_seconds: PositiveFloat = Field(
        default=2,
        description="Short-term average (STA) window length in seconds. "
        "Only used when `model` is `STA/LTA`.",
    )
    lta_seconds: PositiveFloat = Field(
        default=5.0,
        description="Long-term average (LTA) window length in seconds. "
        "Only used when `model` is `STA/LTA`.",
    )
    blinding_window: PositiveFloat = Field(
        default=5,
        description="Blinding window in which no new detection can be set. "
        "Typically the duration of the seismic event."
        "Only used when `model` is `STA/LTA`.",
    )
    signal_transform: SignalTransform = Field(
        default="energy",
        description="Signal transform applied to the waveform before computing "
        "the STA/LTA ratio. `energy` uses the squared amplitude, `absolute` the "
        "absolute amplitude, and `envelope` the amplitude of the analytic signal "
        "(Hilbert envelope).",
    )
    mad_factor: PositiveFloat = Field(
        default=10.0,
        description="Multiplier for the median absolute deviation (MAD) of a "
        "station's STA/LTA trace, used as the phase-picking trigger threshold: "
        "threshold = MAD * mad_factor.",
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
        stream = Stream(tr.to_obspy_trace() for tr in traces)

        char_function_traces = await asyncio.to_thread(
            _compute_characteristic_functions,
            stream,
            self.sta_seconds,
            self.lta_seconds,
            self.signal_transform,
        )

        traces = to_pyrocko_traces(char_function_traces)

        p_traces = [tr for tr in traces if tr.channel.endswith("Z")]
        s_traces = [tr for tr in traces if not tr.channel.endswith("Z")]
        s_traces = _merge_horizontal_components(s_traces)

        annotation_p = StaLtaImage(
            image_function=self.name,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            detection_half_width=self._detection_half_width(),
            traces=p_traces,
            mad_factor=self.mad_factor,
        )
        annotation_s = StaLtaImage(
            image_function=self.name,
            weight=self.weights["S"],
            phase=self.phase_map["S"],
            detection_half_width=self._detection_half_width(),
            traces=s_traces,
            mad_factor=self.mad_factor,
        )
        return [annotation_s, annotation_p]

    def get_blinding(self) -> timedelta:
        """Blinding duration for the image function. Added to padded waveforms.

        Returns:
            timedelta: The blinding duration for the image function.
        """
        return timedelta(seconds=self.blinding_window)

    def get_provided_phases(self) -> tuple[PhaseDescription, ...]:
        """Get the phases provided by the image function.

        Returns:
            tuple[PhaseDescription, ...]: The phases provided by the image function.
        """
        return tuple(self.phase_map.values())

    def _detection_half_width(self) -> float:
        """Half width of the detection window in seconds."""
        return self.blinding_window / 2
