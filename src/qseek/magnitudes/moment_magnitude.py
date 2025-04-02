from __future__ import annotations

import asyncio
import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, NamedTuple

import numpy as np
from pydantic import DirectoryPath, Field, PositiveFloat, PrivateAttr
from pyrocko import gf, io

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
from qseek.utils import (
    CACHE_DIR,
    NSL,
    ChannelSelector,
    ChannelSelectors,
    MeasurementUnit,
    Range,
    _Range,
    time_to_path,
)

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver
    from qseek.models.station import Stations
    from qseek.octree import Octree
    from qseek.waveforms.base import WaveformProvider


logger = logging.getLogger(__name__)

KM = 1e3

_TRACE_SELECTORS: dict[PeakAmplitude, ChannelSelector] = {
    "absolute": ChannelSelectors.All,
    "vertical": ChannelSelectors.Vertical,
    "horizontal": ChannelSelectors.Horizontal,
}


def norm_traces(traces: list[Trace]) -> np.ndarray:
    """Normalizes the traces to their maximum absolute value.

    Args:
        traces (list[Trace]): The traces to normalize.

    Returns:
        np.ndarray: The normalized traces.
    """
    data = np.array([tr.ydata for tr in traces])
    return np.linalg.norm(np.atleast_2d(data), axis=0)


class PeakAmplitudeDefinition(PeakAmplitudesBase):
    nsl_id: list[str] | None = Field(
        default=None,
        pattern=r"^[A-Z0-9]{2}\.[A-Z0-9]{0,5}?\.[A-Z0-9]{0,2}?$",
        description="Network, station, location id.",
    )
    peak_amplitude: PeakAmplitude = Field(
        default="absolute",
        description="The peak amplitude to use.",
    )
    station_epicentral_range: Range = Field(
        default=_Range(min=1 * KM, max=100 * KM),
        description="The epicentral distance range of the stations.",
    )
    frequency_range: Range = Field(
        default=_Range(min=2.0, max=20.0),
        description="The frequency range in Hz to filter the traces.",
    )

    def filter_receivers_by_nsl(self, receivers: Iterable[Receiver]) -> set[Receiver]:
        """Filters the list of receivers based on the NSL ID.

        Args:
            receivers (list[Receiver]): The list of receivers to filter.

        Returns:
            list[Receiver]: The filtered list of receivers.
        """
        if self.nsl_id is None:
            return set(receivers)

        matched_receivers = []
        for nsl in self.nsl_id:
            nsl = NSL.parse(nsl)
            matched_receivers.append([rcv for rcv in receivers if rcv.nsl.match(nsl)])

        return set(itertools.chain.from_iterable(matched_receivers))

    def filter_receivers_by_range(
        self,
        receivers: Iterable[Receiver],
        event: EventDetection,
    ) -> set[Receiver]:
        """Filters the list of receivers based on the distance range.

        Args:
            receivers (Iterable[Receiver]): The list of receivers to filter.
            event (EventDetection): The event detection object.

        Returns:
            set[Receiver]: The filtered set of receivers.
        """
        receivers = [
            rcv
            for rcv in receivers
            if self.station_epicentral_range.inside(rcv.surface_distance_to(event))
        ]
        return set(receivers)


class StationMomentMagnitude(NamedTuple):
    station: NSL
    distance_epi: float
    magnitude: float
    error: float
    peak_amp: float
    snr: float = 0.0


class MomentMagnitude(EventMagnitude):
    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    station_magnitudes: list[StationMomentMagnitude] = Field(
        default_factory=list,
        description="The station moment magnitudes.",
    )
    quantity: MeasurementUnit = Field(
        default="displacement",
        description="The quantity of the traces.",
    )

    @property
    def m0(self) -> float:
        return 10.0 ** (1.5 * (self.average + 10.7)) * 1.0e-7

    @property
    def n_stations(self) -> int:
        """Number of stations used for calculating the moment magnitude."""
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            "Mw": self.average,
            "Mw-error": self.error,
        }

    async def add_traces(
        self,
        store: PeakAmplitudesStore,
        peak_amplitude: PeakAmplitude,
        event: EventDetection,
        receivers: list[Receiver],
        traces: list[list[Trace]],
        noise_padding: float = 0.5,
        min_snr: float = 3.0,
    ) -> None:
        for receiver, rcv_traces in zip(receivers, traces, strict=False):
            try:
                rcv_traces = _TRACE_SELECTORS[peak_amplitude](rcv_traces)
            except (KeyError, AttributeError):
                continue
            if not rcv_traces:
                logger.warning("No traces for peak amplitude %s", receiver.nsl.pretty)
                continue

            station = StationAmplitudes.create(
                receiver=receiver,
                traces=rcv_traces,
                noise_padding=noise_padding,
                event=event,
                measurement="max-amplitude",
            )

            if station.distance_epi > store.max_distance:
                continue
            if station.snr < min_snr:
                logger.debug("%s has low SNR %g", receiver.nsl.pretty, station.snr)
                continue

            try:
                model = await store.get_amplitude_model(
                    source_depth=event.effective_depth,
                    distance=station.distance_epi,
                    n_amplitudes=25,
                    peak_amplitude=peak_amplitude,
                    interpolation="linear",
                )
            except ValueError:
                logger.warning("No modelled amplitude for receiver %s", receiver.nsl)
                continue

            magnitude = model.estimate_magnitude(station.peak)
            error_upper = (
                model.estimate_magnitude(station.peak + station.noise) - magnitude
            )
            error_lower = (
                model.estimate_magnitude(station.peak - station.noise) - magnitude
            )
            if not np.isfinite(error_lower):
                error_lower = error_upper

            station_magnitude = StationMomentMagnitude(
                # quantity=store.quantity,
                station=receiver.nsl,
                distance_epi=station.distance_epi,
                magnitude=magnitude,
                error=(error_upper + abs(error_lower)) / 2,
                peak_amp=station.peak,
                snr=station.snr,
            )
            self.station_magnitudes.append(station_magnitude)

        if not self.station_magnitudes:
            return

        magnitudes = np.array([sta.magnitude for sta in self.station_magnitudes])
        median = np.median(magnitudes)

        self.average = float(median)
        self.error = float(np.median(np.abs(magnitudes - median)))  # MAD


class MomentMagnitudeExtractor(EventMagnitudeCalculator):
    """Moment magnitude calculator from peak amplitudes."""

    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    noise_window: PositiveFloat = Field(
        default=5.0,
        ge=1.5,
        description="Waveforms to extract before P phase arrival. The noise amplitude "
        "is extracted from before the P phase arrival, with a one second padding.",
    )
    seconds_after: PositiveFloat = Field(
        default=4.0,
        description="Waveforms to extract after S phase arrival.",
    )
    taper_seconds: PositiveFloat = Field(
        default=10.0,
        description="Seconds tapering before and after the extraction window."
        " The taper stabalizes the restitution and is cut off from the traces "
        "before the analysis.",
    )
    min_signal_noise_ratio: float = Field(
        default=1.5,
        ge=1.0,
        description="Minimum signal-to-noise ratio for the magnitude estimation. "
        "The noise amplitude is extracted from before the P phase arrival,"
        " with 0.5 s padding.",
    )

    gf_store_dirs: list[DirectoryPath] = Field(
        default=[Path(".")],
        description="The directories of the Pyrocko GF stores.",
    )
    export_mseed: Path | None = Field(
        default=None,
        description="Path to export the processed mseed traces to.",
    )

    models: list[PeakAmplitudeDefinition] = Field(
        default=[PeakAmplitudeDefinition()],
        description="The peak amplitude models to use.",
        min_length=1,
    )

    _stores: list[PeakAmplitudesStore] = PrivateAttr()

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        logger.info("Preparing peak amplitude stores...")

        engine = gf.LocalEngine(store_superdirs=self.gf_store_dirs)
        cache = PeakAmplitudeStoreCache(CACHE_DIR / "peak_amplitudes", engine)
        logger.info("Loading peak amplitude stores...")
        self._stores = [cache.get_store(definition) for definition in self.models]

        octree_depth = octree.location.effective_depth
        for store, definition in zip(self._stores, self.models, strict=True):
            await store.fill_source_depth_range(
                depth_min=octree.depth_bounds.min + octree_depth,
                depth_max=octree.depth_bounds.max + octree_depth,
                depth_delta=definition.source_depth_delta,
            )

    def has_magnitude(self, event: EventDetection) -> bool:
        if not event.magnitudes:
            return False
        return any(type(mag) is MomentMagnitude for mag in event.magnitudes)

    async def add_magnitude(
        self,
        waveform_provider: WaveformProvider,
        event: EventDetection,
    ) -> None:
        moment_magnitude = MomentMagnitude()
        receivers = list(event.receivers)

        for store, definition in zip(self._stores, self.models, strict=True):
            store_receivers = definition.filter_receivers_by_nsl(receivers)
            if not store_receivers:
                continue
            for rcv in store_receivers:
                receivers.remove(rcv)
            store_receivers = definition.filter_receivers_by_range(
                store_receivers, event
            )
            if not store_receivers:
                logger.warning("No receivers in range for peak amplitude store")
                continue
            if not store.source_depth_range.inside(event.effective_depth):
                logger.warning(
                    "Event depth %.1f outside of magnitude store range (%.1f - %.1f).",
                    event.effective_depth,
                    *store.source_depth_range,
                )
                continue

            traces = await event.receivers.get_waveforms_restituted(
                waveform_provider.get_squirrel(),
                receivers=store_receivers,
                quantity=store.quantity,
                seconds_before=self.noise_window,
                seconds_after=self.seconds_after,
                seconds_taper=self.taper_seconds,
                channels=waveform_provider.channel_selector,
                demean=True,
                cut_off_taper=False,
                filter_clipped=True,
            )
            if not traces:
                continue

            if store.frequency_range.min != 0.0:
                for tr in traces:
                    await asyncio.to_thread(
                        tr.bandpass,
                        4,
                        store.frequency_range.min,
                        store.frequency_range.max,
                        demean=False,
                    )
            else:
                for tr in traces:
                    await asyncio.to_thread(
                        tr.lowpass, 4, store.frequency_range.max, demean=False
                    )

            for tr in traces:
                await asyncio.to_thread(
                    tr.chop, tr.tmin + self.taper_seconds, tr.tmax - self.taper_seconds
                )

            if self.export_mseed is not None:
                file_name = self.export_mseed / f"{time_to_path(event.time)}.mseed"
                logger.debug("saving restituted mseed traces to %s", file_name)
                await asyncio.to_thread(io.save, traces, str(file_name))

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
                min_snr=self.min_signal_noise_ratio,
            )

        if not moment_magnitude.magnitude:
            logger.warning("No moment magnitude for event %s", event.time)

        event.add_magnitude(moment_magnitude)
