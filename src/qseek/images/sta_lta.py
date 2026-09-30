from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Annotated, Literal

import numpy as np
from obspy import Stream
from obspy.signal.trigger import trigger_onset
from pydantic import Field, PositiveFloat
from pyrocko.obspy_compat import to_pyrocko_traces
from pyrocko.trace import Trace
from scipy.signal import hilbert
from scipy.stats import median_abs_deviation

from qseek.images.base import (
    ImageFunction,
    ObservedArrival,
    PhaseName,
    Picker,
    WaveformImage,
)
from qseek.utils import PhaseDescription, to_datetime

# The signal transformation, the STA/LTA computation, and the merging of multiple horizontal components
# are based on the implementation in QuakeMigrate[1]: https://github.com/QuakeMigrate/QuakeMigrate
# [1] Winder, T., Bacon, C.A., Smith, J.D., Hudson, T., Greenfield, T. and White, R.S., 2020. QuakeMigrate: a Modular,
# Open-Source Python Package for Automatic Earthquake Detection and Location. In AGU Fall Meeting 2020. AGU.

logger = logging.getLogger(__name__)

SignalTransform = Literal["energy", "absolute", "envelope"]
StaLtaPosition = Literal["centred", "classic"]

# The LTA window is invalid at the start of each trace, blind it with a margin
LTA_BLINDING_MARGIN = 1.2


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

    Both windows end at the evaluated sample. The ratio therefore peaks up to
    `nsta` samples after a phase onset, which delays the image function and biases
    the stacked origin times late. The STA window is part of the LTA window, the
    ratio is limited to `nlta / nsta`.

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


def _centred_sta_lta(signal: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
    """Centred STA/LTA ratio of an already non-negative signal.

    The LTA window ends at the evaluated sample and the STA window starts at the
    following sample. The ratio of a phase onset peaks one sample before the
    onset, it is not delayed by the STA window.

    Args:
        signal (np.ndarray): Non-negative characteristic signal.
        nsta (int): Number of samples in the short-term window.
        nlta (int): Number of samples in the long-term window.

    Returns:
        np.ndarray: STA/LTA ratio, computed in adjacent windows.
    """
    sta = np.cumsum(signal, dtype=np.float64)
    lta = sta.copy()

    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    # Shift the STA window to start after the evaluated sample
    sta[nsta:-nsta] = sta[nsta * 2 :]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad with ones (= null result) where the LTA or STA window is not full.
    sta[: nlta - 1] = 1.0
    lta[: nlta - 1] = 1.0
    sta[-nsta:] = 1.0
    lta[-nsta:] = 1.0

    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    sta[idx] = dtiny

    return sta / lta


def _log_onset(onset: np.ndarray, min_onset_value: float) -> np.ndarray:
    """Clip an STA/LTA onset function and take its natural logarithm.

    The delay-and-sum stack of the log onsets is the logarithm of the geometric
    mean of the onsets, as the coalescence in QuakeMigrate. Noise (STA/LTA = 1)
    maps to 0 and single stations with large ratios do not dominate the stack.

    Args:
        onset (np.ndarray): STA/LTA onset function.
        min_onset_value (float): Minimum onset value before taking the logarithm.

    Returns:
        np.ndarray: Log onset function.
    """
    return np.log(np.clip(onset, min_onset_value, None))


def _merge_horizontal_components(traces: list[Trace]) -> list[Trace]:
    """Combine the horizontal-component onset traces of each station.

    Aligned segments of the components, sharing network/station/location, start
    time and number of samples, are combined as the root-mean-square of their
    STA/LTA onset functions. Segments without an aligned counterpart, e.g. due to
    a data gap in a single component, are kept as single-component traces.
    Where the resulting traces of a station overlap, the longest trace is kept.

    Args:
        traces (list[Trace]): Per-component STA/LTA onset traces.

    Returns:
        list[Trace]: Non-overlapping onset traces per station.
    """
    grouped: dict[tuple[str, str, str], dict[tuple[int, int], list[Trace]]] = {}
    for tr in traces:
        segment = (round(tr.tmin / tr.deltat), tr.ydata.size)
        station_segments = grouped.setdefault((tr.network, tr.station, tr.location), {})
        station_segments.setdefault(segment, []).append(tr)

    merged_traces = []
    for station_segments in grouped.values():
        station_traces = []
        for components in station_segments.values():
            if len(components) == 1:
                station_traces.append(components[0])
                continue
            stacked = np.array([tr.ydata for tr in components])
            rms = np.sqrt(np.sum(stacked**2, axis=0) / len(components))
            merged = components[0].copy()
            merged.set_ydata(rms)
            station_traces.append(merged)

        kept: list[Trace] = []
        for tr in sorted(station_traces, key=lambda tr: tr.ydata.size, reverse=True):
            if any(tr.tmin <= other.tmax and other.tmin <= tr.tmax for other in kept):
                logger.warning(
                    "cannot merge misaligned horizontal components of %s, dropping %s",
                    ".".join(tr.nslc_id[:3]),
                    ".".join(tr.nslc_id),
                )
                continue
            kept.append(tr)
        merged_traces.extend(sorted(kept, key=lambda tr: tr.tmin))

    return merged_traces


def _compute_characteristic_functions(
    stream: Stream,
    sta_seconds: float,
    lta_seconds: float,
    signal_transform: SignalTransform,
    position: StaLtaPosition = "centred",
) -> list[Trace]:
    """Compute the STA/LTA characteristic function.

    For each trace, the waveform is first transformed into a non-negative
    characteristic signal (`signal_transform`), then the short-term and
    long-term average windows are converted from seconds to samples based on
    the sampling rate, and the centred or classic STA/LTA ratio is computed.
    Traces that are shorter than the required LTA window are skipped and a warning is logged.

    Args:
        stream (Stream): containing the input seismic traces
        sta_seconds (float): Duration of the short-term average window in seconds.
        lta_seconds (float): Duration of the long-term average window in seconds.
        signal_transform (SignalTransform): Transform applied to the waveform
            before computing the STA/LTA ratio.
        position (StaLtaPosition): Position of the STA window, `centred` after or
            `classic` overlapping the end of the LTA window.

    Returns:
        list of Traces: A list of traces containing the STA/LTA characteristic
        functions.
    """
    sta_lta = _centred_sta_lta if position == "centred" else _overlapping_sta_lta
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
        tr.data = sta_lta(transformed, sta_samples, lta_samples)
        char_function_traces.append(tr)

    return char_function_traces


class StaLtaPicker(Picker):
    """Pick phase onsets from STA/LTA characteristic functions.

    Triggers are detected on the station's full STA/LTA trace. A trigger turns on
    where the ratio exceeds `median + mad_factor * MAD` of the trace and turns off
    when it falls below half of that excess, `median + mad_factor / 2 * MAD`. The
    pick is the peak of the trigger closest to the modelled arrival within the
    search window, the centred STA/LTA peaks at the phase onset. Peaks before the
    event origin time are rejected.
    """

    mad_factor: PositiveFloat = Field(
        default=5.0,
        description="Trigger threshold above the median of a station's STA/LTA "
        "trace, in multiples of its median absolute deviation (MAD): "
        "threshold = median + MAD * mad_factor.",
    )
    search_window_seconds: PositiveFloat = Field(
        default=5.0,
        description="Total length of the search window in seconds, centered on the"
        " modelled arrival time.",
    )

    def get_trigger_thresholds(self, data: np.ndarray) -> tuple[float, float]:
        """Get the trigger on and off thresholds for a STA/LTA trace.

        Args:
            data (np.ndarray): STA/LTA characteristic function.

        Returns:
            tuple[float, float]: Trigger on and off thresholds.
        """
        median = float(np.median(data))
        mad = float(median_abs_deviation(data))
        return (
            median + self.mad_factor * mad,
            median + self.mad_factor / 2 * mad,
        )

    def pick_trace(
        self,
        trace: Trace,
        phase: PhaseDescription,
        event_time: datetime,
        modelled_arrival: datetime,
    ) -> ObservedArrival | None:
        """Pick the trigger peak closest to the modelled arrival.

        Args:
            trace (Trace): STA/LTA characteristic function trace.
            phase (PhaseDescription): Phase of the observed arrival.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.

        Returns:
            ObservedArrival | None: Picked arrival, None if none found.
        """
        data = trace.ydata.astype(np.float64, copy=False)
        half_window = self.search_window_seconds / 2
        window_tmin = modelled_arrival.timestamp() - half_window
        window_tmax = modelled_arrival.timestamp() + half_window
        if window_tmax < trace.tmin or window_tmin > trace.tmax:
            logger.warning("No data to pick phase arrival %s.", ".".join(trace.nslc_id))
            return None

        threshold_on, threshold_off = self.get_trigger_thresholds(data)
        triggers = np.asarray(
            trigger_onset(data, threshold_on, threshold_off), dtype=int
        ).reshape(-1, 2)
        # A trigger active from the first sample has no observed onset
        triggers = triggers[triggers[:, 0] > 0]

        peak_idx = np.array(
            [on + np.argmax(data[on : off + 1]) for on, off in triggers], dtype=int
        )
        peak_times = trace.tmin + peak_idx * trace.deltat
        valid = (
            (peak_times >= window_tmin)
            & (peak_times <= window_tmax)
            & (peak_times > event_time.timestamp())
        )
        peak_idx = peak_idx[valid]
        peak_times = peak_times[valid]
        if not peak_times.size:
            return None
        if peak_times.size > 1:
            logger.debug(
                "%d triggers found for %s, picking the one closest to the "
                "modelled arrival.",
                peak_times.size,
                ".".join(trace.nslc_id),
            )

        closest = np.argmin(np.abs(peak_times - modelled_arrival.timestamp()))
        return ObservedArrival(
            time=to_datetime(peak_times[closest]),
            detection_value=float(data[peak_idx[closest]]),
            phase=phase,
        )


class StaLta(ImageFunction):
    """STA/LTA analytical characteristic function.

    The image is the natural logarithm of the STA/LTA onset function, clipped at
    `min_onset_value`. Stacking the log onsets yields the logarithm of the
    geometric mean of the onsets, following QuakeMigrate. The semblance and the
    pick detection values are in log units, noise is at 0.
    """

    image: Literal["StaLta"] = "StaLta"

    sta_seconds: PositiveFloat = Field(
        default=0.2,
        description="Short-term average (STA) window length in seconds. A long STA"
        " window flattens the peak of the centred STA/LTA at the phase onset.",
    )
    lta_seconds: PositiveFloat = Field(
        default=1.0,
        description="Long-term average (LTA) window length in seconds.",
    )
    blinding_window: PositiveFloat = Field(
        default=5,
        description="Blinding window in which no new detection can be set. "
        "Typically the duration of the seismic event.",
    )
    position: StaLtaPosition = Field(
        default="centred",
        description="Position of the STA window. `centred` places the STA window"
        " after the LTA window, the ratio peaks at the phase onset. `classic`"
        " overlaps both windows at their end, the ratio peaks up to `sta_seconds`"
        " after the phase onset.",
    )
    min_onset_value: float = Field(
        default=0.4,
        ge=0.01,
        description="Minimum value of the STA/LTA onset function before taking the "
        "logarithm. Limits the influence of low onset values, e.g. in the coda of "
        "strong events, on the stack.",
    )
    signal_transform: SignalTransform = Field(
        default="energy",
        description="Signal transform applied to the waveform before computing "
        "the STA/LTA ratio. `energy` uses the squared amplitude, `absolute` the "
        "absolute amplitude, and `envelope` the amplitude of the analytic signal "
        "(Hilbert envelope).",
    )

    phase_map: dict[PhaseName, str] = Field(
        default={
            "P": "cake:P",
            "S": "cake:S",
        },
        description="Phase mapping from STA/LTA P and S images to "
        "Qseek travel time phases.",
    )
    weights: dict[PhaseName, Annotated[float, Field(strict=True, ge=0.0)]] = Field(
        default={
            "P": 1.0,
            "S": 1.0,
        },
        description="Weights for each phase.",
    )
    picker: StaLtaPicker = Field(
        default_factory=StaLtaPicker,
        description="Picker to use for the image function.",
    )

    async def prepare(self) -> None: ...

    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
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
            self.position,
        )

        traces = to_pyrocko_traces(char_function_traces)

        p_traces = [tr for tr in traces if tr.channel.endswith("Z")]
        s_traces = [tr for tr in traces if not tr.channel.endswith("Z")]
        s_traces = _merge_horizontal_components(s_traces)
        for tr in p_traces + s_traces:
            tr.set_ydata(_log_onset(tr.ydata, self.min_onset_value))

        annotation_p = WaveformImage(
            image_function=self.name,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            detection_half_width=self._detection_half_width(),
            traces=p_traces,
        )
        annotation_s = WaveformImage(
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
        return timedelta(seconds=self.lta_seconds * LTA_BLINDING_MARGIN)

    def get_phases(self) -> tuple[PhaseDescription, ...]:
        """Get the phases provided by the image function.

        Returns:
            tuple[PhaseDescription, ...]: The phases provided by the image function.
        """
        return tuple(self.phase_map.values())

    def _detection_half_width(self) -> float:
        """Half width of the detection window in seconds."""
        return self.blinding_window / 2
