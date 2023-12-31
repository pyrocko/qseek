from __future__ import annotations

import asyncio
import itertools
import logging
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from matplotlib.ticker import FuncFormatter
from pydantic import Field, PositiveFloat, PrivateAttr, computed_field, model_validator
from pyrocko import trace
from typing_extensions import Self

from qseek.features.utils import ChannelSelector, TraceSelectors
from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
)
from qseek.magnitudes.local_magnitude_model import (
    LocalMagnitudeModel,
    ModelName,
    Range,
    StationLocalMagnitude,
)

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver

logger = logging.getLogger(__name__)

# New corrections from NMSOP 3.3
# 10.2312/GFZ.NMSOP-2_IS_3.3, Page 5
WOOD_ANDERSON = trace.PoleZeroResponse(
    poles=[
        -5.49779 + 5.60886j,
        -5.49779 - 5.60886j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

KM = 1e3
MM = 1e3


class WoodAndersonAmplitude(NamedTuple):
    peak_mm: float
    noise_mm: float
    std_noise_mm: float

    @property
    def anr(self) -> float:
        """Amplitude to noise ratio."""
        return self.peak_mm / self.noise_mm

    @classmethod
    def from_traces(
        cls,
        traces: list[Trace],
        noise_traces: list[Trace],
        selector: ChannelSelector,
    ) -> Self:
        peak_amp = max(np.abs(trace.ydata).max() for trace in selector(traces))
        noise_amp = max(np.abs(trace.ydata).max() for trace in selector(noise_traces))
        std_noise = max(np.std(trace.ydata) for trace in selector(noise_traces))

        return cls(
            peak_mm=peak_amp * MM,
            noise_mm=noise_amp * MM,
            std_noise_mm=std_noise * MM,
        )


class StationAmplitudes(NamedTuple):
    station_nsl: tuple[str, str, str]
    amplitudes_horizontal: WoodAndersonAmplitude
    amplitudes_vertical: WoodAndersonAmplitude
    distance_epi: float
    distance_hypo: float

    def in_range(
        self,
        epi_range: Range | None = None,
        hypo_range: Range | None = None,
    ) -> bool:
        if not epi_range and not hypo_range:
            return True
        if epi_range:
            return epi_range.min <= self.distance_epi <= epi_range.max
        if hypo_range:
            return hypo_range.min <= self.distance_hypo <= hypo_range.max
        return False

    @classmethod
    def from_receiver(
        cls,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
        noise_padding: float = 3.0,
    ) -> Self:
        time_arrival = min(receiver.get_arrivals_time_window()).timestamp()
        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=time_arrival - noise_padding, inplace=False)
            for tr in traces
        ]
        return cls(
            station_nsl=receiver.nsl,
            amplitudes_horizontal=WoodAndersonAmplitude.from_traces(
                traces=traces,
                noise_traces=noise_traces,
                selector=TraceSelectors.Horizontal,
            ),
            amplitudes_vertical=WoodAndersonAmplitude.from_traces(
                traces=traces,
                noise_traces=noise_traces,
                selector=TraceSelectors.Vertical,
            ),
            distance_hypo=receiver.distance_to(event),
            distance_epi=receiver.surface_distance_to(event),
        )


class LocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    station_amplitudes: list[StationAmplitudes] = []
    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )

    _station_magnitudes: list[StationLocalMagnitude] = PrivateAttr([])
    _model: LocalMagnitudeModel = PrivateAttr()

    def add_receiver(
        self,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
    ) -> None:
        try:
            self.station_amplitudes.append(
                StationAmplitudes.from_receiver(receiver, traces, event)
            )
            self._station_magnitudes.clear()
        except Exception as exc:
            logger.warning(
                "Could not calculate station amplitudes for receiver %s: %s",
                receiver.nsl_pretty,
                exc,
            )

    @property
    def name(self) -> str:
        return f"{self.model_name}"

    @model_validator(mode="after")
    def load_model(self) -> Self:
        self._model = LocalMagnitudeModel.get_subclass_by_name(self.model)()
        return self

    def set_model(self, model: ModelName) -> None:
        self.model = model
        self._model = LocalMagnitudeModel.get_subclass_by_name(model)()
        self._station_magnitudes.clear()

    @property
    def station_magnitudes(self) -> list[StationLocalMagnitude]:
        if not self._station_magnitudes:
            self._station_magnitudes = self._model.get_station_magnitudes(
                self.station_amplitudes
            )
            logger.debug("Calculated magnitude from %d stations", self.n_observations)
        return self._station_magnitudes

    @property
    def magnitudes(self) -> np.ndarray:
        return np.array([sta.magnitude for sta in self.station_magnitudes])

    @property
    def magnitude_errors(self) -> np.ndarray:
        return np.array([sta.magnitude_error for sta in self.station_magnitudes])

    @property
    def n_observations(self) -> int:
        return len(self._station_magnitudes)

    @property
    def model_name(self) -> str:
        return self._model.model_name()

    @computed_field
    @property
    def average(self) -> float:
        return float(np.average(self.magnitudes))

    @computed_field
    @property
    def error(self) -> float:
        return float(np.average(self.magnitudes + self.magnitude_errors)) - self.average

    @computed_field
    @property
    def error_weighted(self) -> float:
        return (
            float(
                np.average(
                    self.magnitudes + self.magnitude_errors,
                    weights=1.0 / self.magnitude_errors,
                )
            )
            - self.average_weighted
        )

    @computed_field
    @property
    def average_weighted(self) -> float:
        return float(np.average(self.magnitudes, weights=1.0 / self.magnitude_errors))

    @computed_field
    @property
    def median(self) -> float:
        return float(np.median(self.magnitudes))

    def csv_row(self) -> dict[str, float]:
        return {
            f"ML_{self.name}": self.average,
            f"ML_error_{self.name}": self.error,
        }

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        station_distances_hypo = np.array(
            [sta.distance_hypo for sta in self._station_magnitudes]
        )

        fig = plt.figure()
        ax = fig.gca()
        ax.errorbar(
            station_distances_hypo,
            self.magnitudes,
            yerr=[sta.magnitude_error for sta in self._station_magnitudes],
            marker="o",
            mec="k",
            mfc="k",
            ms=2,
            ecolor=(0.0, 0.0, 0.0, 0.1),
            capsize=1,
            ls="none",
        )
        ax.axhline(
            self.median,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Median $M_L$ {self.median:.2f}",
        )
        ax.axhline(
            self.average,
            color="k",
            linestyle="dotted",
            alpha=0.5,
            label=rf"Average $M_L$ {self.average:.2f} $\pm${self.error:.2f}",
        )
        ax.axhline(
            self.average_weighted,
            color="k",
            linestyle="-",
            alpha=0.5,
            label=f"Weighted Average $M_L$ {self.average_weighted:.2f} "
            rf"$\pm${self.error_weighted:.2f}",
        )
        ax.set_xlabel("Distance to Hypocenter [km]")
        ax.set_ylabel("$M_L$")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
        ax.grid(alpha=0.3)
        ax.legend(title=f"Estimator: {self._model.model_name()}", loc="lower right")
        ax.text(
            0.05,
            0.05,
            f"{self.n_observations} Stations",
            transform=ax.transAxes,
            alpha=0.5,
        )
        plt.show()


class LocalMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    seconds_before: PositiveFloat = 10.0
    seconds_after: PositiveFloat = 10.0
    padding_seconds: PositiveFloat = 10.0
    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )

    async def add_magnitude(self, squirrel: Squirrel, event: EventDetection) -> None:
        # Check if event already has a local magnitude
        for magnitude in event.magnitudes:
            if isinstance(magnitude, LocalMagnitude):
                magnitude.set_model(self.model)
                logger.debug("Skipping event %s, already has a local magnitude", event)
                return

        traces = await event.receivers.get_waveforms_restituted(
            squirrel,
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            seconds_fade=self.padding_seconds,
            cut_off_fade=False,
            quantity="displacement",
            phase=None,
            remove_clipped=True,
        )
        if not traces:
            logger.warning("No restituted traces found for event %s", event.time)
            return

        wood_anderson_traces = [
            await asyncio.to_thread(
                tr.transfer,
                transfer_function=WOOD_ANDERSON,
                tfade=self.padding_seconds,
                cut_off_fading=True,
                demean=True,
                invert=False,
            )
            for tr in traces
        ]

        local_magnitude = LocalMagnitude(model=self.model)

        for nsl, traces in itertools.groupby(
            wood_anderson_traces, key=lambda tr: tr.nslc_id[:3]
        ):
            local_magnitude.add_receiver(
                receiver=event.receivers.get_receiver(nsl),
                traces=list(traces),
                event=event,
            )

        if not np.isfinite(local_magnitude.average):
            logger.warning("Local magnitude is NaN, skipping event %s", event.time)
            return

        logger.info(
            "Ml %.1f (±%.2f) for event %s",
            local_magnitude.average,
            local_magnitude.error,
            event.time,
        )
        event.add_magnitude(local_magnitude)
