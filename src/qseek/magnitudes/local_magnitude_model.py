from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Type

import numpy as np

if TYPE_CHECKING:
    from qseek.magnitudes.local_magnitude import StationAmplitudes

KM = 1e3
MM = 1e3
MM2NM = 1e6

Component = Literal["horizontal", "vertical"]
WOOD_ANDERSON_CONSTANT = 2080.0

# All models from Bormann (2012) https://doi.org/10.2312/GFZ.NMSOP-2_DS_3.1
# Page 5


class Range(NamedTuple):
    min: float
    max: float


class StationLocalMagnitude(NamedTuple):
    station_nsl: tuple[str, str, str]
    magnitude: float
    magnitude_error: float
    peak_amp_mm: float
    distance_epi: float
    distance_hypo: float


class LocalMagnitudeModel:
    epicentral_range: ClassVar[Range | None] = None
    hypocentral_range: ClassVar[Range | None] = None

    author: ClassVar[str] = "Unknown"
    component: ClassVar[Component] = "horizontal"

    @classmethod
    def get_subclass_by_name(cls, name: str) -> Type[LocalMagnitudeModel]:
        subclasses = {sub.model_name(): sub for sub in cls.__subclasses__()}
        if name in subclasses:
            return subclasses[name]
        raise ValueError(f"Model {name} not found. Choose from {', '.join(subclasses)}")

    @classmethod
    def model_name(cls) -> str:
        """Get the name of the model in kebab case."""
        return re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()

    @classmethod
    def model_names(cls) -> tuple[str, ...]:
        return tuple(sub.model_name() for sub in cls.__subclasses__())

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        """Get the amplitude attenuation for a given distance, known as -log10(A0)."""
        raise NotImplementedError

    def get_local_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        """Get the magnitude from a given amplitude."""
        return np.log10(amplitude) + self.get_amp_attenuation(
            distance_hypo / KM, distance_epi / KM
        )

    def get_station_magnitudes(
        self, amplitudes: list[StationAmplitudes]
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
        for sta in amplitudes:
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

            # TODO: Remove clipped stations

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
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class IaspeiSouthernCalifornia(LocalMagnitudeModel):
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "horizontal"

    def get_local_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        amp = amplitude * MM2NM  # mm to nm
        return (
            np.log10(amp / WOOD_ANDERSON_CONSTANT)
            + 1.11 * np.log10(distance_hypo / KM)
            + 0.00189 * (distance_hypo / KM)
            - 2.09
        )


class EasternNorthAmerica(LocalMagnitudeModel):
    author = "Kim (1998)"

    epicentral_range = Range(100.0 * KM, 800.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * np.log10(dist_epi_km) - 0.22


class Albania(LocalMagnitudeModel):
    author = "Muco and Minga (1991)"

    epicentral_range = Range(10.0 * KM, 600.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * np.log10(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(LocalMagnitudeModel):
    author = "Stange (2006)"

    hypocentral_range = Range(10.0 * KM, 1000.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(LocalMagnitudeModel):
    author = "Greenhalgh and Singh (1986)"

    epicentral_range = Range(40.0 * KM, 700.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * np.log10(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(LocalMagnitudeModel):
    author = "Alsaker et al. (1991)"

    hypocentral_range = Range(0.0 * KM, 1500.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * np.log10(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


class IcelandAskja(LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.4406 * np.log10(dist_hypo_km / 17) - 0.003 * (dist_hypo_km - 17) + 2


class IcelandBardabunga(LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.2534 * np.log10(dist_hypo_km / 17) - 0.0032 * (dist_hypo_km - 17) + 2


class IcelandAskjaBardabungaCombined(LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1999 * np.log10(dist_hypo_km / 17) - 0.0016 * (dist_hypo_km - 17) + 2


class IcelandReykjanes(LocalMagnitudeModel):
    author = "Greenfield et al. (2022)"

    hypocentral_range = Range(0.0 * KM, 40.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.6902 * np.log10(dist_hypo_km / 17) - 0.0318 * (dist_hypo_km - 17) + 2


class Azores(LocalMagnitudeModel):
    author = "Gongora et al. (2004)"

    epicentral_range = Range(10.0 * KM, 800.0 * KM)
    component = "horizontal"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.89 * np.log10(dist_epi_km / 100) + 0.00256 * (dist_epi_km - 100) + 3


ModelName = Literal[LocalMagnitudeModel.model_names()]
