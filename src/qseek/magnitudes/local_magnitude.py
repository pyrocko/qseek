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
        description="Average local magnitude.",
    )
    average_weighted: float = Field(
        default=0.0,
        description="Weighted average local magnitude.",
    )
    median: float = Field(
        default=0.0,
        description="Median local magnitude.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of local magnitude.",
    )

    @classmethod
    async def from_model(
        cls,
        model: LocalMagnitudeModel,
        event: EventDetection,
        receivers: list[Receiver],
        grouped_traces: list[list[Trace]],
    ) -> Self:
        self = cls(model=model.model_name())

        for traces, receiver in zip(grouped_traces, receivers, strict=True):
            station_magnitude = model.get_station_magnitude(
                event=event,
                traces=traces,
                receiver=receiver,
            )
            if station_magnitude is None:
                continue
            self.station_magnitudes.append(station_magnitude)

        magnitudes = self.magnitudes
        magnitude_errors = np.array(
            [sta.magnitude_error for sta in self.station_magnitudes]
        )
        weights = 1.0 / magnitude_errors

        self.average = float(np.average(magnitudes))
        self.average_weighted = float(np.average(self.magnitudes, weights=weights))
        self.median = float(np.median(magnitudes))
        self.error = float(np.average(magnitudes + magnitude_errors)) - self.average

        return self

    @property
    def magnitudes(self) -> np.ndarray:
        return np.array([sta.magnitude for sta in self.station_magnitudes])

    @property
    def n_observations(self) -> int:
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            f"ML_{self.model}": self.average,
            f"ML_error_{self.model}": self.error,
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
            label=f"Weighted Average $M_L$ {self.average_weighted:.2f}",
        )
        ax.set_xlabel("Distance to Hypocenter [km]")
        ax.set_ylabel("$M_L$")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
        ax.grid(alpha=0.3)
        ax.legend(title=f"Estimator: {self.model}", loc="lower right")
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

    _model: LocalMagnitudeModel = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        self._model = LocalMagnitudeModel.get_subclass_by_name(self.model)()
        return self

    async def add_magnitude(self, squirrel: Squirrel, event: EventDetection) -> None:
        model = self._model

        cut_off_fade = model.max_amplitude not in (
            "wood-anderson",
            "wood-anderson-old",
        )

        traces = await event.receivers.get_waveforms_restituted(
            squirrel,
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            seconds_fade=self.padding_seconds,
            cut_off_fade=cut_off_fade,
            quantity=model.restitution_quantity,
            phase=None,
            remove_clipped=True,
        )
        if not traces:
            logger.warning("No restituted traces found for event %s", event.time)
            return

        if model.max_amplitude == "wood-anderson":
            traces = [
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
        elif model.max_amplitude == "wood-anderson-old":
            traces = [
                await asyncio.to_thread(
                    tr.transfer,
                    transfer_function=WOOD_ANDERSON_OLD,
                    tfade=self.padding_seconds,
                    cut_off_fading=True,
                    demean=True,
                    invert=False,
                )
                for tr in traces
            ]

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
