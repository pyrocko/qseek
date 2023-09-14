from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal, Union

import numpy as np
from pydantic import BaseModel, Field
from pyrocko import trace
from pyrocko.squirrel.error import NotAvailable

from lassie.features.base import EventFeature, FeatureExtractor, ReceiverFeature
from lassie.features.utils import ChannelSelector, ChannelSelectors

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.models.detection import EventDetection, Receiver

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


class StationMagnitude(ReceiverFeature):
    feature: Literal["StationMagnitude"] = "StationMagnitude"
    estimator: str

    local_magnitude: float
    peak_velocity_mm: float


class LocalMagnitude(EventFeature):
    feature: Literal["LocalMagnitude"] = "LocalMagnitude"

    estimator: str
    median: float
    mean: float
    std: float
    n_stations: int


class LocalMagnitudeModel(BaseModel):
    name: Literal["local-magnitude-estimator"] = "local-magnitude-estimator"

    epicentral_range: ClassVar[tuple[float, float] | None] = None
    hypocentral_range: ClassVar[tuple[float, float] | None] = None

    trace_selector: ClassVar[ChannelSelector] = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        raise NotImplementedError

    def _get_max_amplitude_mm(self, traces: list[Trace]) -> float:
        return max(np.abs(trace.ydata).max() for trace in self.trace_selector(traces))

    def _is_distance_valid(self, dist_hypo: float, dist_epi: float) -> bool:
        epi_range = self.epicentral_range
        hypo_range = self.hypocentral_range
        if epi_range and epi_range[0] <= dist_epi <= epi_range[1]:
            return True
        if hypo_range and hypo_range[0] <= dist_hypo <= hypo_range[1]:
            return True
        return False

    def calculate_magnitude(
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
        if not self._is_distance_valid(dist_hypo, dist_epi):
            return None

        log_amp_0 = self.get_amp_0(dist_hypo / KM, dist_epi / KM)
        amp_max = self._get_max_amplitude_mm(traces)
        local_magnitude = np.log(amp_max) + log_amp_0

        if not np.isfinite(local_magnitude):
            return None

        return StationMagnitude(
            estimator=self.name,
            local_magnitude=local_magnitude,
            peak_velocity_mm=amp_max,
        )


class SouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    name: Literal["southern-california"] = "southern-california"
    hypocentral_range = (10.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class IASPEISouthernCalifornia(LocalMagnitudeModel):
    """After Hutton&Boore (1987)"""

    name: Literal["iaspei-southern-california"] = "iaspei-southern-california"
    hypocentral_range = (10.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def calculate_magnitude(
        self, event: EventDetection, receiver: Receiver, traces: list[Trace]
    ) -> StationMagnitude | None:
        dist_hypo = event.distance_to(receiver) / KM
        try:
            amp_max = self._get_max_amplitude_mm(traces)
        except KeyError as exc:
            logger.exception(exc)
            return None

        amp_max *= 1e6  # To nm
        local_magnitude = (
            np.log10(amp_max) + 1.11 * np.log10(dist_hypo) + 0.00189 * dist_hypo - 2.09
        )
        if not np.isfinite(local_magnitude):
            return None
        return StationMagnitude(
            estimator=self.name,
            local_magnitude=local_magnitude,
            peak_velocity_mm=amp_max / 1000000,
        )


class EasternNorthAmerica(LocalMagnitudeModel):
    """After Kim (1998)"""

    name: Literal["eastern-north-america"] = "eastern-north-america"
    epicentral_range = (100.0 * KM, 800.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * np.log(dist_epi_km) - 0.22


class Albania(LocalMagnitudeModel):
    """After Muco&Minga (1991)"""

    name: Literal["albania"] = "albania"
    epicentral_range = (10.0 * KM, 600.0 * KM)
    trace_selector = ChannelSelectors.Horizontal

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * np.log(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(LocalMagnitudeModel):
    """After Stange (2006)"""

    name: Literal["south-west-germany"] = "south-west-germany"
    hypocentral_range = (10.0 * KM, 1000.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(LocalMagnitudeModel):
    """After Greenhalgh&Singh (1986)"""

    name: Literal["south-australia"] = "south-australia"
    epicentral_range = (40.0 * KM, 700.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * np.log(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(LocalMagnitudeModel):
    """After Alsaker et al. (1991)"""

    name: Literal["norway-fennoskandia"] = "norway-fennoskandia"
    hypocentral_range = (0.0 * KM, 1500.0 * KM)
    trace_selector = ChannelSelectors.Vertical

    def get_amp_0(self, dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * np.log(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


class LocalMagnitudeExtractor(FeatureExtractor):
    feature: Literal["LocalMagnitude"] = "LocalMagnitude"

    seconds_before: float = 3.0
    seconds_after: float = 10.0
    window_slack: float = 10.0
    estimator: Union[
        IASPEISouthernCalifornia,
        SouthernCalifornia,
        EasternNorthAmerica,
        Albania,
        SouthWestGermany,
        SouthAustralia,
        NorwayFennoscandia,
    ] = Field(
        default=IASPEISouthernCalifornia(),
        discriminator="name",
    )

    async def add_features(self, squirrel: Squirrel, event: EventDetection) -> None:
        local_magnitudes: list[StationMagnitude] = []
        for receiver in event.receivers:
            try:
                traces = receiver.get_waveforms_restituted(
                    squirrel,
                    seconds_before=self.seconds_before - self.window_slack,
                    seconds_after=self.seconds_after + self.window_slack,
                    quantity="velocity",
                )
            except NotAvailable:
                logger.error("cannot get responses for %s", receiver.pretty_nsl)
                continue

            restituted_traces = [
                tr.transfer(transfer_function=WOOD_ANDERSON) for tr in traces
            ]

            for tr in restituted_traces:
                tr.chop(
                    float(tr.tmin) + self.window_slack,
                    float(tr.tmax) - self.window_slack,
                    inplace=True,
                )

            magnitude = self.estimator.calculate_magnitude(
                event, receiver, restituted_traces
            )
            if magnitude is None:
                continue

            receiver.add_feature(magnitude)
            local_magnitudes.append(magnitude)

        magnitudes = [mag.local_magnitude for mag in local_magnitudes]
        if not magnitudes:
            return

        local_magnitude = LocalMagnitude(
            estimator=self.estimator.name,
            median=float(np.median(magnitudes)),
            mean=float(np.mean(magnitudes)),
            std=float(np.std(magnitudes)),
            n_stations=len(magnitudes),
        )
        print(event.time, local_magnitude)
        event.magnitude = local_magnitude.median
        event.magnitude_type = "local"
        event.features.add_feature(local_magnitude)
