from __future__ import annotations

from datetime import timedelta

import numpy as np
from pydantic import BaseModel, PrivateAttr
from pyrocko.trace import Trace

from qseek.models.detection import EventDetection
from qseek.pre_processing.resample import resample
from qseek.utils import _NSL, NSL


def _correlation_matrix(
    traces_a: list[Trace],
    traces_b: list[Trace],
) -> tuple[np.ndarray, np.ndarray]:
    """Get the time shift and normalized cross-correlation coefficient for every trace pair.

    Vectorized as one batched FFT cross-correlation: for stations i, j the full
    linear cross-correlation is ``irfft(rfft(a_i) * conj(rfft(b_j)))``. Its peak
    normalized by the traces' energies gives the coefficient; the peak's lag,
    combined with each trace's absolute ``tmin``, gives the time shift.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(time_shift, cc_values)``, where
        ``time_shift[i, j]`` is ``traces_a[i]``'s arrival time minus ``traces_b[j]``'s
        arrival time (seconds) at the correlation peak, and ``cc_values[i, j]`` is the
        peak normalized cross-correlation coefficient.
    """
    if len(traces_a) == 0 or len(traces_b) == 0:
        raise ValueError("Both trace lists must be non-empty.")
    if len(traces_a) != len(traces_b):
        raise ValueError(
            "Both trace lists must have the same number of traces (common stations)."
        )
    if [tr.nslc_id[:3] for tr in traces_a] != [tr.nslc_id[:3] for tr in traces_b]:
        raise ValueError(
            "Both trace lists must have the same station order (common stations)."
        )

    deltat = traces_a[0].deltat

    data_a = [tr.ydata.astype(np.float32) for tr in traces_a]
    data_b = [tr.ydata.astype(np.float32) for tr in traces_b]

    len_a = max(data.size for data in data_a)
    len_b = max(data.size for data in data_b)
    n_fft = len_a + len_b - 1

    matrix_a = np.zeros((len(data_a), n_fft), dtype=np.float32)
    for i_station, data in enumerate(data_a):
        matrix_a[i_station, : data.size] = data

    matrix_b = np.zeros((len(data_b), n_fft), dtype=np.float32)
    for i_station, data in enumerate(data_b):
        matrix_b[i_station, : data.size] = data

    energy_a = np.sum(matrix_a**2, axis=1)
    energy_b = np.sum(matrix_b**2, axis=1)
    norm = np.sqrt(np.outer(energy_a, energy_b))

    spectrum_a = np.fft.rfft(matrix_a, axis=1)
    spectrum_b = np.fft.rfft(matrix_b, axis=1)
    cross_spectrum = spectrum_a[:, None, :] * np.conj(spectrum_b[None, :, :])
    correlation = np.fft.irfft(cross_spectrum, n=n_fft, axis=-1)

    peak_index = correlation.argmax(axis=-1)
    peak_value = np.take_along_axis(correlation, peak_index[..., None], axis=-1)[..., 0]

    cc_values = np.divide(
        peak_value,
        norm,
        out=np.zeros_like(peak_value),
        where=norm > 0.0,
    ).astype(np.float32)

    # The circular correlation index wraps for negative lags: indices below
    # len_a are non-negative sample lags, the rest fold back from -1.
    lag_samples = np.where(peak_index < len_a, peak_index, peak_index - n_fft)

    tmin_a = np.array([tr.tmin for tr in traces_a])
    tmin_b = np.array([tr.tmin for tr in traces_b])
    time_shift = ((tmin_a[:, None] - tmin_b[None, :]) + lag_samples * deltat).astype(
        np.float32
    )

    return time_shift, cc_values


class EventCorrelationPair(BaseModel):
    event_a: EventDetection
    event_b: EventDetection

    stations: list[NSL]
    phase: str

    _cc_values: np.ndarray = PrivateAttr()
    _time_shift: np.ndarray = PrivateAttr()

    @property
    def origin_time_differences(self) -> timedelta:
        return self.event_b.time - self.event_a.time

    @classmethod
    async def calculate(
        cls,
        event_a: EventDetection,  # Maybe replace by light event model
        event_b: EventDetection,
        traces_a: list[Trace],
        traces_b: list[Trace],
        phase: str,
    ) -> EventCorrelationPair:
        """Calculate the cross-correlation between two events' traces for a given phase.

        Args:
            event_a (EventDetection): The first event.
            event_b (EventDetection): The second event.
            traces_a (list[Trace]): The traces for the first event.
            traces_b (list[Trace]): The traces for the second event.
            phase (str): The phase for which to calculate cross-correlation.

        Raises:
            ValueError: If there are no common stations between the two events.

        Returns:
            EventCorrelationPair: The cross-correlation results for the two events.
        """
        traces_a_by_nsl = {_NSL.parse(tr.nslc_id[:3]): tr for tr in traces_a}
        traces_b_by_nsl = {_NSL.parse(tr.nslc_id[:3]): tr for tr in traces_b}
        common_stations = sorted(set(traces_a_by_nsl) & set(traces_b_by_nsl))
        n_stations = len(common_stations)

        if not n_stations:
            raise ValueError(
                f"No common stations between events {event_a.uid} and {event_b.uid}"
            )

        # Align both trace lists to the same station order
        traces_a = [traces_a_by_nsl[nsl] for nsl in common_stations]
        traces_b = [traces_b_by_nsl[nsl] for nsl in common_stations]

        target_deltat = max(tr.deltat for tr in (*traces_a, *traces_b))
        traces_a = resample(traces_a, target_deltat, demean=True)
        traces_b = resample(traces_b, target_deltat, demean=True)

        time_shift, cc_values = _correlation_matrix(traces_a, traces_b)

        instance = cls(
            event_a=event_a,
            event_b=event_b,
            stations=common_stations,
            phase=phase,
        )
        instance._cc_values = cc_values
        instance._time_shift = time_shift
        return instance


class EventCorrelationCollection(BaseModel):
    cross_correlations: list[EventCorrelationPair]

    @property
    def n_pairs(self) -> int:
        return len(self.cross_correlations)

    async def add_pair(self, pair: EventCorrelationPair) -> None:
        """Add a new cross-correlation pair to the collection."""
        self.cross_correlations.append(pair)
