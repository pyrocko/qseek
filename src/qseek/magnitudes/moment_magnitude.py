from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pydantic import DirectoryPath, Field, PositiveFloat, PrivateAttr
from pyrocko import gf

from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
    StationAmplitudes,
)
from qseek.magnitudes.moment_magnitude_store import (
    PeakAmplitude,
    PeakAmplitudesBase,
    PeakAmplitudesStore,
    PeakAmplitudeStoreCache,
)
from qseek.utils import CACHE_DIR, NSL, ChannelSelector, ChannelSelectors

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver


logger = logging.getLogger(__name__)

KM = 1e3

_TRACE_SELECTORS: dict[PeakAmplitude, ChannelSelector] = {
    "absolute": ChannelSelectors.All,
    "vertical": ChannelSelectors.Vertical,
    "horizontal": ChannelSelectors.Horizontal,
}


def norm_traces(traces: list[Trace]) -> np.ndarray:
    """
    Normalizes the traces to their maximum absolute value.

    Args:
        traces (list[Trace]): The traces to normalize.

    Returns:
        np.ndarray: The normalized traces.
    """
    return np.linalg.norm(np.array([tr.ydata for tr in traces]), axis=0)


class PeakAmplitudeDefinition(PeakAmplitudesBase):
    nsl_id: str | None = Field(
        default=None,
        pattern=r"^[A-Z0-9]{2}\.[A-Z0-9]{0,5}?\.[A-Z0-9]{0,2}?$",
        description="Network, station, location id.",
    )
    peak_amplitude: PeakAmplitude = Field(
        default="absolute",
        description="The peak amplitude to use.",
    )

    def filter_receivers(self, receivers: list[Receiver]) -> list[Receiver]:
        """
        Filters the list of receivers based on the NSL ID.

        Args:
            receivers (list[Receiver]): The list of receivers to filter.

        Returns:
            list[Receiver]: The filtered list of receivers.
        """
        if self.nsl_id is None:
            return receivers
        nsl = NSL.parse(self.nsl_id)
        return [rcv for rcv in receivers if rcv.nsl.match(nsl)]


class StationMomentMagnitude(NamedTuple):
    distance_epi: float
    magnitude: float
    error: float

    peak: float


class MomentMagnitude(EventMagnitude):
    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    average: float = Field(
        default=0.0,
        description="Average moment magnitude.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of moment magnitude.",
    )
    std: float = Field(
        default=0.0,
        description="Standard deviation of moment magnitude.",
    )

    stations_magnitudes: list[StationMomentMagnitude] = Field(
        default_factory=list,
        description="The station moment magnitudes.",
    )

    @property
    def n_stations(self) -> int:
        """
        Number of stations used for calculating the moment magnitude.
        """
        return len(self.stations_magnitudes)

    async def add_traces(
        self,
        store: PeakAmplitudesStore,
        peak_amplitude: PeakAmplitude,
        event: EventDetection,
        receivers: list[Receiver],
        traces: list[list[Trace]],
        noise_padding: float = 3.0,
    ) -> None:
        def get_magnitude(
            measured_amplitude: float,
            modelled_amplitude: float,
        ) -> float:
            return (
                np.log10(measured_amplitude / modelled_amplitude)
                + store.reference_magnitude
            )

        for receiver, rcv_traces in zip(receivers, traces, strict=False):
            try:
                rcv_traces = _TRACE_SELECTORS[peak_amplitude](rcv_traces)
            except KeyError:
                continue
            if not rcv_traces:
                logger.warning("No traces for peak amplitude %s", receiver.nsl.pretty)
                continue

            station_amplitudes = StationAmplitudes.create(
                receiver=receiver,
                traces=rcv_traces,
                noise_padding=noise_padding,
                event=event,
            )

            if station_amplitudes.anr < 1.0:
                logger.warning("Station %s has bad ANR", receiver.nsl.pretty)
                continue
            if store.max_distance < station_amplitudes.distance_epi:
                continue

            try:
                modelled_amplitude = await store.get_amplitude(
                    source_depth=event.depth,
                    distance=station_amplitudes.distance_epi,
                    n_amplitudes=25,
                    peak_amplitude=peak_amplitude,
                    interpolation="linear",
                )
            except ValueError:
                logger.warning("No modelled amplitude for receiver %s", receiver.nsl)
                continue

            magnitude = get_magnitude(
                station_amplitudes.peak,
                modelled_amplitude.amplitude_median,
            )
            error_upper = (
                get_magnitude(
                    station_amplitudes.peak + station_amplitudes.noise,
                    modelled_amplitude.amplitude_median,
                )
                - magnitude
            )
            error_lower = (
                get_magnitude(
                    station_amplitudes.peak - station_amplitudes.noise,
                    modelled_amplitude.amplitude_median,
                )
                - magnitude
            )

            station_magnitude = StationMomentMagnitude(
                distance_epi=station_amplitudes.distance_epi,
                magnitude=magnitude,
                error=(error_upper + error_lower) / 2,
                peak=station_amplitudes.peak,
            )
            self.stations_magnitudes.append(station_magnitude)

        if not self.stations_magnitudes:
            return

        magnitudes = np.array([sta.magnitude for sta in self.stations_magnitudes])
        errors = np.array([sta.error for sta in self.stations_magnitudes])

        self.average = float(np.average(magnitudes))
        self.error = float(np.average(errors))
        self.std = float(np.std(magnitudes))


class MomentMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    seconds_before: PositiveFloat = 10.0
    seconds_after: PositiveFloat = 10.0
    padding_seconds: PositiveFloat = 10.0

    gf_store_dirs: list[DirectoryPath] = [Path(".")]

    models: list[PeakAmplitudeDefinition] = Field(
        default_factory=list,
        description="The peak amplitude models to use.",
    )

    _stores: list[PeakAmplitudesStore] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        if not self.models:
            raise ValueError("No peak amplitude models specified.")

        engine = gf.LocalEngine(store_superdirs=self.gf_store_dirs)
        cache = PeakAmplitudeStoreCache(CACHE_DIR / "peak_amplitudes", engine)
        logger.info("Loading peak amplitude stores...")
        self._stores = [cache.get_store(definition) for definition in self.models]

    async def add_magnitude(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        moment_magnitude = MomentMagnitude()

        receivers = list(event.receivers)
        for store, definition in zip(self._stores, self.models, strict=True):
            store_receivers = definition.filter_receivers(receivers)
            if not store_receivers:
                continue
            for rcv in store_receivers:
                receivers.remove(rcv)

            traces = await event.receivers.get_waveforms_restituted(
                squirrel,
                quantity=store.quantity,
                seconds_before=self.seconds_before,
                seconds_after=self.seconds_after,
                demean=True,
                cut_off_fade=True,
                seconds_fade=self.padding_seconds,
            )
            if not traces:
                logger.warning("No restituted traces found for event %s", event.time)
                return

            for tr in traces:
                tr.highpass(4, store.frequency_range.min)
                tr.lowpass(4, store.frequency_range.max)

            grouped_traces = []
            receivers = []
            for nsl, grp_traces in itertools.groupby(
                traces, key=lambda tr: tr.nslc_id[:3]
            ):
                grouped_traces.append(list(grp_traces))
                receivers.append(event.receivers.get_receiver(nsl))
                await moment_magnitude.add_traces(
                    store=store,
                    event=event,
                    receivers=receivers,
                    traces=grouped_traces,
                    peak_amplitude=definition.peak_amplitude,
                )

        if not moment_magnitude.average:
            logger.warning("No moment magnitude found for event %s", event.time)
            return

        event.add_magnitude(moment_magnitude)
