from __future__ import annotations

import asyncio
import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import Field, PositiveFloat, PrivateAttr, model_validator
from pyrocko import io

from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
    StationAmplitudes,
)
from qseek.magnitudes.local_magnitude_models import (
    WOOD_ANDERSON,
    WOOD_ANDERSON_OLD,
    WOOD_ANDERSON_PIDSA,
    CustomLocalMagnitudeModel,
    LocalMagnitudeModel,
    ModelName,
    StationLocalMagnitude,
)
from qseek.utils import time_to_path

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver
    from qseek.models.station import StationInventory
    from qseek.waveforms.providers import WaveformProvider

logger = logging.getLogger(__name__)


class EventLocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    model: str = Field(
        default="iaspei-southern-california",
        description="The attenuation model to use for calculating the local magnitude.",
    )
    station_magnitudes: list[StationLocalMagnitude] = Field(default=[])

    @property
    def n_stations(self) -> int:
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            f"ML-{self.model}": self.average,
            f"ML-error-{self.model}": self.error,
            f"ML-n-stations-{self.model}": self.n_stations,
        }

    @classmethod
    def from_station_magnitudes(
        cls,
        model_name: str,
        station_magnitudes: list[StationLocalMagnitude],
        max_station_std: float = 3.0,
        min_stations: int = 3,
    ) -> Self:
        ml = cls(model=model_name)

        if not station_magnitudes:
            raise ValueError("No station magnitudes provided")

        std = np.std([sta.magnitude for sta in station_magnitudes])
        unclean_mean = np.mean([sta.magnitude for sta in station_magnitudes])

        for mag in station_magnitudes.copy():
            if np.abs(mag.magnitude - unclean_mean) > max_station_std * std:
                logger.warning(
                    "%s magnitude removed due to high std",
                    mag.station.pretty,
                )
                station_magnitudes.remove(mag)

        if len(station_magnitudes) < min_stations:
            raise ValueError(
                "Not enough station magnitudes available for local magnitude "
                "calculation after removing outliers."
            )

        ml.station_magnitudes = station_magnitudes
        magnitudes = np.array([sta.magnitude for sta in ml.station_magnitudes])
        station_errors = np.array([sta.error for sta in ml.station_magnitudes])
        n = len(magnitudes)
        median = np.median(magnitudes)

        ml.average = float(median)
        # Combine inter-station scatter (MAD) and per-station measurement noise,
        # both normalized by sqrt(N) to give the precision of the median estimate.
        mad = np.median(np.abs(magnitudes - median))
        ml.error = float(
            np.sqrt(
                (mad / np.sqrt(n)) ** 2 + (np.median(station_errors) / np.sqrt(n)) ** 2
            )
        )
        return ml


class LocalMagnitude(EventMagnitudeCalculator):
    """Local magnitude calculator for different regional models."""

    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    noise_window: PositiveFloat = Field(
        default=5.0,
        ge=1.5,
        description="Waveforms to extract before P phase arrival. The noise amplitude "
        "is extracted from before the P phase arrival, with 0.5 s padding.",
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
        description="Minimum signal-to-noise ratio for the local magnitude estimation. "
        "The noise amplitude is extracted from before the P phase arrival,"
        " with 0.5 s padding.",
    )
    max_station_std: float = Field(
        default=3.0,
        ge=0.0,
        description="Maximum allowed standard deviation of a station magnitude before "
        "it is disregarded relative to the network magnitude.",
    )

    model: ModelName | CustomLocalMagnitudeModel = Field(
        default="iaspei-southern-california",
        description="The amplitude attenuation model to "
        "use for calculating the local magnitude, or a custom local magnitude model.",
    )

    export_mseed: Path | None = Field(
        default=None,
        description="Path to export the processed mseed traces to.",
    )

    _model: LocalMagnitudeModel = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if isinstance(self.model, str):
            self._model = LocalMagnitudeModel.get_subclass_by_name(self.model)()
        elif isinstance(self.model, CustomLocalMagnitudeModel):
            self._model = self.model.get_model()
        else:
            raise ValueError(
                "model must be either a ModelName or a CustomLocalMagnitudeModel"
            )
        return self

    def csv_header(self) -> list[str]:
        return [
            f"ML-{self._model.model_name()}",
            f"ML-error-{self._model.model_name()}",
        ]

    def has_magnitude(self, event: EventDetection) -> bool:
        for mag in event.magnitudes:
            if type(mag) is EventLocalMagnitude and mag.model == self.model:
                return True
        return False

    async def get_magnitude(
        self,
        waveform_provider: WaveformProvider,
        stations: StationInventory,
        event: EventDetection,
    ) -> EventLocalMagnitude:
        """Calculate the local magnitude for the given event.

        The local magnitude is calculated by extracting the restituted waveforms for
        the event, applying the appropriate transfer function for the model, and
        then calculating the station magnitudes and average magnitude using the model.

        Args:
            waveform_provider: The waveform provider to use for extracting the
                waveforms.
            stations: The station inventory to use for calculating the station
                magnitudes.
            event: The event for which to calculate the local magnitude.
        """
        model = self._model

        traces = await event.receivers.get_waveforms_restituted(
            waveform_provider,
            stations,
            seconds_before=self.noise_window,
            seconds_after=self.seconds_after,
            seconds_taper=self.taper_seconds,
            quantity=model.restitution_quantity,
            channels=waveform_provider.channel_selector,
            exclude_nsls=self.exclude_stations,
            phase=None,
            cut_off_taper=False,
            filter_clipped=True,
            want_incomplete=False,
        )
        if not traces:
            raise ValueError("No traces found for event")

        if model.max_amplitude.startswith("wood-anderson"):
            # This is a circus: Different versions of the wood-anderson
            # transfer function are used
            if model.max_amplitude == "wood-anderson":
                transfer_function = WOOD_ANDERSON
            elif model.max_amplitude == "wood-anderson-old":
                transfer_function = WOOD_ANDERSON_OLD
            elif model.max_amplitude == "wood-anderson-2800":
                transfer_function = WOOD_ANDERSON_PIDSA
            else:
                raise ValueError(f"Unknown wood-anderson model: {model.max_amplitude}")

            traces = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        tr.transfer,
                        transfer_function=transfer_function,
                        freqlimits=(0.1, 1.0, 0.40 / tr.deltat, 0.45 / tr.deltat),
                        tfade=self.taper_seconds,
                        cut_off_fading=False,
                        demean=True,
                        invert=False,
                    )
                    for tr in traces
                ]
            )

        if model.highpass_freq is not None:
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        tr.highpass,
                        order=4,
                        corner=model.highpass_freq,
                    )
                    for tr in traces
                ]
            )
        if model.lowpass_freq is not None:
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        tr.lowpass,
                        order=4,
                        corner=model.lowpass_freq,
                    )
                    for tr in traces
                ]
            )

        await asyncio.gather(
            *[
                asyncio.to_thread(
                    tr.chop,
                    tr.tmin + self.taper_seconds,
                    tr.tmax - self.taper_seconds,
                )
                for tr in traces
            ]
        )

        for tr in traces:
            tr.ydata -= np.mean(tr.ydata)

        if self.export_mseed is not None:
            file_name = self.export_mseed / f"{time_to_path(event.time)}.mseed"
            logger.debug("saving restituted mseed traces to %s", file_name)
            await asyncio.to_thread(io.save, traces, str(file_name))

        sorted_traces = sorted(traces, key=lambda tr: tr.nslc_id[:3])
        station_magnitudes = []
        for nsl, grp_traces in itertools.groupby(
            sorted_traces,
            key=lambda tr: tr.nslc_id[:3],
        ):
            try:
                sta_mag = self.get_station_magnitude(
                    model=model,
                    event=event,
                    receiver=event.receivers.get_receiver(nsl),
                    traces=list(grp_traces),
                )
            except ValueError as exc:
                logger.debug(
                    "Could not calculate station magnitude for receiver %s: %s",
                    nsl,
                    exc,
                )
                continue

            if sta_mag.snr < self.min_signal_noise_ratio:
                logger.debug(
                    "Excluding station %s from local magnitude calculation due to low "
                    "SNR: %.2f",
                    nsl,
                    sta_mag.snr,
                )
                continue
            station_magnitudes.append(sta_mag)

        return EventLocalMagnitude.from_station_magnitudes(
            model_name=model.model_name(),
            station_magnitudes=station_magnitudes,
            max_station_std=self.max_station_std,
            min_stations=self.min_stations,
        )

    def get_station_magnitude(
        self,
        model: LocalMagnitudeModel,
        event: EventDetection,
        receiver: Receiver,
        traces: list[Trace],
    ) -> StationLocalMagnitude:
        """Calculate the local magnitude for a given event and receiver.

        Args:
            model (LocalMagnitudeModel): The local magnitude model to use for the
                calculation.
            event (EventDetection): The event to calculate the magnitude for.
            receiver (Receiver): The seismic station to calculate the magnitude for.
            traces (list[Trace]): The traces to calculate the magnitude for.

        Returns:
            StationLocalMagnitude: The calculated magnitude or None if the magnitude
                could not be determined.
        """
        try:
            traces = model.get_amplitude_traces(traces)
        except (KeyError, AttributeError) as exc:
            raise ValueError(
                f"Could not get {model.component} component for receiver "
                f"{receiver.nsl.pretty}"
            ) from exc
        if not traces:
            raise ValueError(
                f"No traces found for receiver {receiver.nsl.pretty} and component "
                f"{model.component}"
            )

        sta_amp = StationAmplitudes.create(
            receiver=receiver,
            traces=traces,
            event=event,
            measurement=model.peak_measurement,
            station_depth_only=model.station_depth_only,
        )

        nsl_pretty = receiver.nsl.pretty_str(strip=True)
        if model.epicentral_range and not model.epicentral_range.inside(
            sta_amp.distance_epi
        ):
            raise ValueError(f"Receiver {nsl_pretty} is outside the epicentral range")
        if model.hypocentral_range and not model.hypocentral_range.inside(
            sta_amp.distance_hypo
        ):
            raise ValueError(f"Receiver {nsl_pretty} is outside the hypocentral range")

        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = model.get_magnitude(
                sta_amp.peak_amp,
                sta_amp.distance_hypo,
                sta_amp.distance_epi,
                receiver.nsl,
            )
            mag_error_upper = model.get_magnitude(
                sta_amp.peak_amp + sta_amp.noise,
                sta_amp.distance_hypo,
                sta_amp.distance_epi,
                receiver.nsl,
            )
            mag_error_lower = model.get_magnitude(
                sta_amp.peak_amp - sta_amp.noise,
                sta_amp.distance_hypo,
                sta_amp.distance_epi,
                receiver.nsl,
            )

        if not np.isfinite(magnitude):
            raise ValueError(
                f"Could not calculate magnitude for receiver {nsl_pretty}"
                f" due to non-finite magnitude value"
            )

        if not np.isfinite(mag_error_lower):
            mag_error_lower = magnitude - (mag_error_upper - magnitude)

        return StationLocalMagnitude(
            station=sta_amp.station_nsl,
            magnitude=float(magnitude),
            error=float(
                ((mag_error_upper - magnitude) + abs(magnitude - mag_error_lower)) / 2
            ),
            peak_amp=sta_amp.peak_amp,
            snr=sta_amp.snr,
            distance_epi=sta_amp.distance_epi,
            distance_hypo=sta_amp.distance_hypo,
        )
