from __future__ import annotations

import asyncio
import itertools
import logging
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, NamedTuple, Union

import numpy as np
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, Field, PositiveFloat, PrivateAttr, computed_field
from pyrocko import trace
from typing_extensions import Self

from qseek.features.utils import ChannelSelector, ChannelSelectors
from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
)

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver

# All models from Bormann and Dewey (2014) https://doi.org/10.2312/GFZ.NMSOP-2_IS_3.3
# Page 5
logger = logging.getLogger(__name__)

Component = Literal["horizontal", "vertical"]

WOOD_ANDERSON = trace.PoleZeroResponse(
    poles=[
        -5.49779 - 5.60886j,
        -5.49779 + 5.60886j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

WOOD_ANDERSON = trace.PoleZeroResponse(
    poles=[
        -6.283 - 4.7124j,
        -6.283 + 4.7124j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

KM = 1e3
MM = 1e3
MM2NM = 1e6


class Range(NamedTuple):
    min: float
    max: float


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
                selector=ChannelSelectors.Horizontal,
            ),
            amplitudes_vertical=WoodAndersonAmplitude.from_traces(
                traces=traces,
                noise_traces=noise_traces,
                selector=ChannelSelectors.Vertical,
            ),
            distance_hypo=receiver.distance_to(event),
            distance_epi=receiver.surface_distance_to(event),
        )


class StationLocalMagnitude(NamedTuple):
    station_nsl: tuple[str, str, str]
    magnitude: float
    magnitude_error: float
    peak_amp_mm: float
    distance_epi: float
    distance_hypo: float


class LocalMagnitudeModel(BaseModel):
    model: Literal["local-magnitude-model"] = "local-magnitude-model"

    epicentral_range: ClassVar[Range | None] = None
    hypocentral_range: ClassVar[Range | None] = None

    component: ClassVar[Component] = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        """Get the amplitude attenuation for a given distance."""
        raise NotImplementedError

    def get_local_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        """Get the magnitude from a given amplitude."""
        return np.log10(amplitude) + self.get_amp_attenuation(
            distance_hypo / KM, distance_epi / KM
        )

    def get_station_magnitudes(
        self, stations: list[StationAmplitudes]
    ) -> list[StationLocalMagnitude]:
        """Calculate the local magnitude for a given event and receiver.

        Args:
            event (EventDetection): The event to calculate the magnitude for.
            receiver (Receiver): The seismic station to calculate the magnitude for.
            traces (list[Trace]): The traces to calculate the magnitude for.

        Returns:
            StationMagnitude | None: The calculated magnitude or None if the magnitude.
        """
        station_magnitudes = []
        for sta in stations:
            if not sta.in_range(self.epicentral_range, self.hypocentral_range):
                continue

            amps = (
                sta.amplitudes_horizontal
                if self.component == "horizontal"
                else sta.amplitudes_vertical
            )

            # Discard stations with no amplitude or low ANR
            if not amps.peak_mm or amps.anr < 1.0:
                continue

            with np.errstate(divide="ignore"):
                local_magnitude = self.get_local_magnitude(
                    amps.peak_mm, sta.distance_hypo, sta.distance_epi
                )
                magnitude_error_upper = self.get_local_magnitude(
                    amps.peak_mm + amps.noise_mm, sta.distance_hypo, sta.distance_epi
                )
                magnitude_error_lower = self.get_local_magnitude(
                    amps.peak_mm - amps.noise_mm, sta.distance_hypo, sta.distance_epi
                )

            if not np.isfinite(local_magnitude):
                continue

            if not np.isfinite(magnitude_error_lower):
                magnitude_error_lower = local_magnitude - (
                    magnitude_error_upper - local_magnitude
                )

            magnitude = StationLocalMagnitude(
                station_nsl=sta.station_nsl,
                magnitude=local_magnitude,
                magnitude_error=(magnitude_error_upper - magnitude_error_lower) / 2,
                peak_amp_mm=amps.peak_mm,
                distance_epi=sta.distance_epi,
                distance_hypo=sta.distance_hypo,
            )
            station_magnitudes.append(magnitude)
        return station_magnitudes


class SouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    model: Literal["southern-california"] = "southern-california"
    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class IASPEISouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    model: Literal["iaspei-southern-california"] = "iaspei-southern-california"
    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "horizontal"

    def get_local_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        amp = amplitude * MM2NM  # mm to nm
        return (
            np.log10(amp / WOOD_ANDERSON.constant)
            + 1.11 * np.log10(distance_hypo / KM)
            + 0.00189 * (distance_hypo / KM)
            - 2.09
        )


class EasternNorthAmerica(LocalMagnitudeModel):
    """After Kim (1998)"""

    model: Literal["eastern-north-america"] = "eastern-north-america"
    epicentral_range = Range(100.0 * KM, 800.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * np.log10(dist_epi_km) - 0.22


class Albania(LocalMagnitudeModel):
    """After Muco&Minga (1991)"""

    model: Literal["albania"] = "albania"
    epicentral_range = Range(10.0 * KM, 600.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * np.log10(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(LocalMagnitudeModel):
    """After Stange (2006)"""

    model: Literal["south-west-germany"] = "south-west-germany"
    hypocentral_range = Range(10.0 * KM, 1000.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(LocalMagnitudeModel):
    """After Greenhalgh&Singh (1986)"""

    model: Literal["south-australia"] = "south-australia"
    epicentral_range = Range(40.0 * KM, 700.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * np.log10(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(LocalMagnitudeModel):
    """After Alsaker et al. (1991)"""

    model: Literal["norway-fennoskandia"] = "norway-fennoskandia"
    hypocentral_range = Range(0.0 * KM, 1500.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * np.log10(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


LocalMagnitudeType = Annotated[
    Union[
        IASPEISouthernCalifornia,
        SouthernCalifornia,
        EasternNorthAmerica,
        Albania,
        SouthWestGermany,
        SouthAustralia,
        NorwayFennoscandia,
    ],
    Field(
        ...,
        discriminator="model",
    ),
]


class LocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    station_amplitudes: list[StationAmplitudes] = []

    _station_magnitudes: list[StationLocalMagnitude] = PrivateAttr([])
    _model: LocalMagnitudeModel = PrivateAttr(default_factory=IASPEISouthernCalifornia)

    def add_receiver(
        self,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
    ) -> None:
        self.station_amplitudes.append(
            StationAmplitudes.from_receiver(receiver, traces, event)
        )

    def set_model(self, model: LocalMagnitudeModel) -> None:
        self._model = model
        self.calculate()

    def calculate(self) -> None:
        self._station_magnitudes = self._model.get_station_magnitudes(
            self.station_amplitudes
        )

    @property
    def station_magnitudes(self) -> list[StationLocalMagnitude]:
        return self._station_magnitudes

    @property
    def magnitudes(self) -> np.ndarray:
        return np.array([sta.magnitude for sta in self.station_magnitudes])

    @property
    def magnitude_errors(self) -> np.ndarray:
        return np.array([sta.magnitude_error for sta in self.station_magnitudes])

    @property
    def n_magnitudes(self) -> int:
        return len(self._station_magnitudes)

    @property
    def model_name(self) -> str:
        return self._model.model

    @computed_field
    @property
    def average(self) -> float:
        return float(np.average(self.magnitudes))

    @computed_field
    @property
    def average_weighted(self) -> float:
        return float(np.average(self.magnitudes, weights=1 / self.magnitude_errors))

    @computed_field
    @property
    def median(self) -> float:
        return float(np.median(self.magnitudes))

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
            label=f"Average $M_L$ {self.average:.2f}",
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
        ax.legend(title=f"Estimator: {self._model.model}", loc="lower right")
        ax.text(
            0.05,
            0.05,
            f"{self.n_magnitudes} Stations",
            transform=ax.transAxes,
            alpha=0.5,
        )
        plt.show()


class LocalMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    seconds_before: PositiveFloat = 10.0
    seconds_after: PositiveFloat = 10.0
    padding_seconds: PositiveFloat = 10.0
    model: LocalMagnitudeType = Field(
        default_factory=IASPEISouthernCalifornia,
        description="The estimator to use for calculating the local magnitude.",
    )

    async def add_magnitude(self, squirrel: Squirrel, event: EventDetection) -> None:
        # p.enable()

        traces = await event.receivers.get_waveforms_restituted(
            squirrel,
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            seconds_fade=self.padding_seconds,
            cut_off_fade=False,
            quantity="displacement",
            phase=None,
        )

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

        local_magnitude = LocalMagnitude()
        local_magnitude.set_model(self.model)

        for nsl, traces in itertools.groupby(
            wood_anderson_traces, key=lambda tr: tr.nslc_id[:3]
        ):
            local_magnitude.add_receiver(
                receiver=event.receivers.get_receiver(nsl),
                traces=list(traces),
                event=event,
            )

        local_magnitude.calculate()
        local_magnitude.plot()
        event.add_magnitude(local_magnitude)
