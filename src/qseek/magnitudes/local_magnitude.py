from __future__ import annotations

import asyncio
import itertools
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.ticker import FuncFormatter
from pydantic import Field, PositiveFloat, PrivateAttr, model_validator
from typing_extensions import Self

from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
)
from qseek.magnitudes.local_magnitude_model import (
    WOOD_ANDERSON,
    WOOD_ANDERSON_OLD,
    LocalMagnitudeModel,
    ModelName,
    StationLocalMagnitude,
)

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver

logger = logging.getLogger(__name__)

KM = 1e3
MM = 1e3


class LocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )
    station_magnitudes: list[StationLocalMagnitude] = []
    average: float = Field(
        default=0.0,
        description="The network's local magnitude, as median of"
        " all station magnitudes.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of local magnitude, as median absolute deviation.",
    )

    @classmethod
    async def from_model(
        cls,
        model: LocalMagnitudeModel,
        event: EventDetection,
        receivers: list[Receiver],
        grouped_traces: list[list[Trace]],
        min_snr: float = 3.0,
    ) -> Self:
        self = cls(model=model.model_name())

        for traces, receiver in zip(grouped_traces, receivers, strict=True):
            station_magnitude = model.get_station_magnitude(
                event=event,
                traces=traces,
                receiver=receiver,
                min_snr=min_snr,
            )
            if station_magnitude is None:
                continue
            self.station_magnitudes.append(station_magnitude)

        if not self.station_magnitudes:
            return self

        median = np.median(self.magnitudes)
        self.average = float(median)
        self.error = float(np.median(np.abs(self.magnitudes - median)))  # MAD

        return self

    @property
    def magnitudes(self) -> np.ndarray:
        return np.array([sta.magnitude for sta in self.station_magnitudes])

    @property
    def n_stations(self) -> int:
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            f"ML-{self.model}": self.average,
            f"ML-error-{self.model}": self.error,
        }

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        station_distances_hypo = np.array(
            [sta.distance_hypo for sta in self.station_magnitudes]
        )

        fig = plt.figure()
        ax = fig.gca()
        ax.errorbar(
            station_distances_hypo,
            self.magnitudes,
            yerr=[sta.magnitude_error for sta in self.station_magnitudes],
            marker="o",
            mec="k",
            mfc="k",
            ms=2,
            ecolor=(0.0, 0.0, 0.0, 0.1),
            capsize=1,
            ls="none",
        )
        ax.axhline(
            self.average,
            color="k",
            linestyle="dotted",
            alpha=0.5,
            label=rf"Median $M_L$ {self.average:.2f} $\pm${self.error:.2f}",
        )
        ax.set_xlabel("Distance to Hypocenter [km]")
        ax.set_ylabel("$M_L$")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
        ax.grid(alpha=0.3)
        ax.legend(title=f"Estimator: {self.model}", loc="lower right")
        ax.text(
            0.05,
            0.05,
            f"{self.n_stations} Stations",
            transform=ax.transAxes,
            alpha=0.5,
        )
        plt.show()


class LocalMagnitudeExtractor(EventMagnitudeCalculator):
    """Local magnitude calculator for different regional models."""

    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    seconds_before: PositiveFloat = Field(
        default=2.0,
        ge=1.0,
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

    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )

    _model: LocalMagnitudeModel = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        self._model = LocalMagnitudeModel.get_subclass_by_name(self.model)()
        return self

    def has_magnitude(self, event: EventDetection) -> bool:
        for mag in event.magnitudes:
            if type(mag) is LocalMagnitude and mag.model == self.model:
                return True
        return False

    async def add_magnitude(self, squirrel: Squirrel, event: EventDetection) -> None:
        model = self._model

        traces = await event.receivers.get_waveforms_restituted(
            squirrel,
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            seconds_fade=self.taper_seconds,
            quantity=model.restitution_quantity,
            cut_off_fade=False,
            phase=None,
            filter_clipped=True,
        )
        if not traces:
            logger.warning("No restituted traces found for event %s", event.time)
            return

        if model.highpass_freq is not None:
            for tr in traces:
                tr.highpass(order=4, corner=model.highpass_freq)
        if model.lowpass_freq is not None:
            for tr in traces:
                tr.lowpass(order=4, corner=model.lowpass_freq)

        if model.max_amplitude == "wood-anderson":
            traces = [
                await asyncio.to_thread(
                    tr.transfer,
                    transfer_function=WOOD_ANDERSON,
                    freqlimits=(0.5, 1.0, 100.0, 200.0),
                    tfade=self.taper_seconds,
                    cut_off_fading=True,
                    demean=True,
                    invert=False,
                )
                for tr in traces
            ]
        elif model.max_amplitude == "wood-anderson-old":
            traces = [
                await asyncio.to_thread(
                    tr.transfer,
                    transfer_function=WOOD_ANDERSON_OLD,
                    freqlimits=(0.5, 1.0, 100.0, 200.0),
                    tfade=self.taper_seconds,
                    cut_off_fading=True,
                    demean=True,
                    invert=False,
                )
                for tr in traces
            ]
        else:
            for tr in traces:
                tr.chop(
                    tr.tmin + self.taper_seconds,
                    tr.tmax - self.taper_seconds,
                    inplace=True,
                )

        grouped_traces = []
        receivers = []
        for nsl, grp_traces in itertools.groupby(traces, key=lambda tr: tr.nslc_id[:3]):
            grouped_traces.append(list(grp_traces))
            receivers.append(event.receivers.get_receiver(nsl))

        local_magnitude = await LocalMagnitude.from_model(
            model=model,
            grouped_traces=grouped_traces,
            receivers=receivers,
            event=event,
            min_snr=self.min_signal_noise_ratio,
        )

        if not np.isfinite(local_magnitude.average):
            logger.warning("Local magnitude is NaN, skipping event %s", event.time)
            return

        event.add_magnitude(local_magnitude)
