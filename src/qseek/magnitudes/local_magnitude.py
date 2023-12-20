from __future__ import annotations

import itertools
import logging
from math import log10
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Union

import numpy as np
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, Field, PositiveFloat
from pyrocko import trace

from qseek.features.utils import ChannelSelector, ChannelSelectors
from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
    StationMagnitude,
)

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver

# From Bormann and Dewey (2014) https://doi.org/10.2312/GFZ.NMSOP-2_IS_3.3
# Page 5
logger = logging.getLogger(__name__)

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


class LocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    estimator: str

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def add_station(self, station_magnitude: StationMagnitude) -> None:
        self.stations.append(station_magnitude)

    @property
    def station_distances_epi(self) -> np.ndarray:
        return np.array([sta.distance_epi for sta in self.stations])

    @property
    def station_distances_hypo(self) -> np.ndarray:
        return np.array([sta.distance_epi for sta in self.stations])

    @property
    def station_magnitudes(self) -> np.ndarray:
        return np.array([sta.magnitude for sta in self.stations])

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ax.errorbar(
            self.station_distances_hypo,
            self.station_magnitudes,
            yerr=[sta.magnitude_error for sta in self.stations],
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
        ax.legend(title=f"Estimator: {self.estimator}", loc="lower right")
        ax.text(
            0.05,
            0.05,
            f"{self.n_stations} Stations",
            transform=ax.transAxes,
            alpha=0.5,
        )
        plt.show()


class LocalMagnitudeModel(BaseModel):
    name: Literal["local-magnitude-estimator"] = "local-magnitude-estimator"

    epicentral_range: ClassVar[tuple[float, float] | None] = None
    hypocentral_range: ClassVar[tuple[float, float] | None] = None

    trace_selector: ClassVar[ChannelSelector] = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        """Get the amplitude correction for a given hypocentral and epicentral
        distance.
        """
        raise NotImplementedError

    def _get_max_amplitude_mm(self, traces: list[Trace]) -> float:
        """Get the maximum amplitude in mm from a list of restituted traces.

        Args:
            traces (list[Trace]): The traces to get the maximum amplitude from.

        Returns:
            float: The maximum amplitude in mm.
        """
        return (
            max(np.abs(trace.ydata).max() for trace in self.trace_selector(traces)) * MM
        )

    def _in_distance_range(self, dist_hypo: float, dist_epi: float) -> bool:
        epi_range = self.epicentral_range
        hypo_range = self.hypocentral_range
        if epi_range and epi_range[0] <= dist_epi <= epi_range[1]:
            return True
        if hypo_range and hypo_range[0] <= dist_hypo <= hypo_range[1]:
            return True
        return False

    def get_noise_amplitude(
        self, receiver: Receiver, traces: list[Trace], padding: float = 2.0
    ) -> float:
        """Get the maximum noise ampltiude in mm from a list of traces.

        Args:
            traces (list[Trace]): The traces to get the noise level from.

        Returns:
            float: The noise level.
        """
        tmax_noise = min(receiver.get_arrivals_time_window()).timestamp() - padding
        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=tmax_noise, inplace=False) for tr in traces
        ]
        return self._get_max_amplitude_mm(noise_traces)

    def get_noise_std(
        self, receiver: Receiver, traces: list[Trace], padding: float = 2.0
    ) -> float:
        tmax_noise = min(receiver.get_arrivals_time_window()).timestamp() - padding
        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=tmax_noise, inplace=False) for tr in traces
        ]
        return max(np.std(trace.ydata) for trace in self.trace_selector(noise_traces))

    def calculate(
        self,
        event: EventDetection,
        receiver: Receiver,
        traces: list[Trace],
    ) -> StationMagnitude | None:
        """Calculate the local magnitude for a given event and receiver.

        Args:
            event (EventDetection): The event to calculate the magnitude for.
            receiver (Receiver): The seismic station to calculate the magnitude for.
            traces (list[Trace]): The traces to calculate the magnitude for.

        Returns:
            StationMagnitude | None: The calculated magnitude or None if the magnitude.
        """
        dist_hypo = event.distance_to(receiver)
        dist_epi = event.surface_distance_to(receiver)
        if not self._in_distance_range(dist_hypo, dist_epi):
            return None

        noise_std = self.get_noise_std(receiver, traces)
        amp_max = self._get_max_amplitude_mm(traces)
        amp_noise = self.get_noise_amplitude(receiver, traces)

        if amp_max < 2 * noise_std:
            return None

        log_amp_0 = self.get_amp_0(dist_hypo / KM, dist_epi / KM)
        with np.errstate(divide="ignore"):
            local_magnitude = log10(amp_max) + log_amp_0
            magnitude_error_upper = log10(amp_max + amp_noise) + log_amp_0
            magnitude_error_lower = log10(amp_max - amp_noise) + log_amp_0

        if not np.isfinite(local_magnitude):
            return None

        if not np.isfinite(magnitude_error_lower):
            magnitude_error_lower = local_magnitude - (
                magnitude_error_upper - local_magnitude
            )

        return StationMagnitude(
            station_nsl=receiver.nsl,
            magnitude=local_magnitude,
            magnitude_error=(magnitude_error_upper - magnitude_error_lower) / 2,
            peak_amp_mm=amp_max,
            distance_epi=dist_epi,
            distance_hypo=dist_hypo,
        )


class SouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    name: Literal["southern-california"] = "southern-california"
    hypocentral_range = (10.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * log10(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class IASPEISouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    name: Literal["iaspei-southern-california"] = "iaspei-southern-california"
    hypocentral_range = (10.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def calculate(
        self, event: EventDetection, receiver: Receiver, traces: list[Trace]
    ) -> StationMagnitude | None:
        amp_max = self._get_max_amplitude_mm(traces)
        amp_noise = self.get_noise_amplitude(receiver, traces)
        noise_std = self.get_noise_std(receiver, traces)

        if amp_max < 2 * noise_std:
            return None

        def model(amp: float, dist_hyp: float) -> float:
            amp *= MM2NM  # mm to nm
            return log10(amp) + 1.11 * log10(dist_hyp) + 0.00189 * dist_hyp - 2.09

        dist_hypo = event.distance_to(receiver) / KM
        with np.errstate(divide="ignore"):
            local_magnitude = model(amp_max, dist_hypo)
            magnitude_error_upper = model(amp_max + amp_noise, dist_hypo)
            magnitude_error_lower = model(amp_max - amp_noise, dist_hypo)

        if not np.isfinite(local_magnitude):
            return None

        if not np.isfinite(magnitude_error_lower):
            magnitude_error_lower = local_magnitude - (
                magnitude_error_upper - local_magnitude
            )

        return StationMagnitude(
            station_nsl=receiver.nsl,
            magnitude=local_magnitude,
            magnitude_error=(magnitude_error_upper - magnitude_error_lower) / 2,
            peak_amp_mm=amp_max,
            distance_epi=event.surface_distance_to(receiver),
            distance_hypo=event.distance_to(receiver),
        )


class EasternNorthAmerica(LocalMagnitudeModel):
    """After Kim (1998)"""

    name: Literal["eastern-north-america"] = "eastern-north-america"
    epicentral_range = (100.0 * KM, 800.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * log10(dist_epi_km) - 0.22


class Albania(LocalMagnitudeModel):
    """After Muco&Minga (1991)"""

    name: Literal["albania"] = "albania"
    epicentral_range = (10.0 * KM, 600.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * log10(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(LocalMagnitudeModel):
    """After Stange (2006)"""

    name: Literal["south-west-germany"] = "south-west-germany"
    hypocentral_range = (10.0 * KM, 1000.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * log10(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(LocalMagnitudeModel):
    """After Greenhalgh&Singh (1986)"""

    name: Literal["south-australia"] = "south-australia"
    epicentral_range = (40.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * log10(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(LocalMagnitudeModel):
    """After Alsaker et al. (1991)"""

    name: Literal["norway-fennoskandia"] = "norway-fennoskandia"
    hypocentral_range = (0.0 * KM, 1500.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * log10(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


EstimatorType = Annotated[
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
        discriminator="name",
    ),
]


class LocalMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    seconds_before: PositiveFloat = 10.0
    seconds_after: PositiveFloat = 10.0
    padding_seconds: PositiveFloat = 10.0
    estimator: EstimatorType = Field(
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
            tr.transfer(
                transfer_function=WOOD_ANDERSON,
                tfade=self.padding_seconds,
                cut_off_fading=True,
                demean=True,
                invert=False,
            )
            for tr in traces
        ]

        # p.disable()
        # p.dump_stats("local_magnitude.prof")
        # trace.snuffle(wood_anderson_traces, markers=event.get_pyrocko_markers())

        station_magnitudes: list[StationMagnitude] = []
        for nsl, traces in itertools.groupby(
            wood_anderson_traces, key=lambda tr: tr.nslc_id[:3]
        ):
            receiver = event.receivers.get_receiver(nsl)
            magnitude = self.estimator.calculate(event, receiver, list(traces))
            if magnitude is None:
                continue
            station_magnitudes.append(magnitude)

        if not station_magnitudes:
            return

        local_magnitude = LocalMagnitude(
            estimator=self.estimator.name,
            stations=station_magnitudes,
        )
        event.add_magnitude(local_magnitude)
