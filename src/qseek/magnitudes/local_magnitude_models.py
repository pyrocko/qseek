from __future__ import annotations

import logging
import re
from typing import ClassVar, Literal, NamedTuple, Self, Type

import numpy as np
from pydantic import Field, PrivateAttr, model_validator
from pyrocko.trace import PoleZeroResponse, Trace

from qseek.base import Model
from qseek.magnitudes.base import PeakMeasurement
from qseek.utils import NSL, ChannelSelector, ChannelSelectors, MeasurementUnit

logger = logging.getLogger(__name__)

KM = 1e3
MM = 1e3
UM = 1e6
M2NM = 1e9

Component = Literal[
    "all",
    "all-abs",
    "horizontal-abs",
    "horizontal-avg",
    "vertical",
    "horizontal-separate",
    "north-east-separate",
]

MaxAmplitudeQuantity = Literal[
    "counts",
    "acceleration",
    "velocity",
    "displacement",
    "wood-anderson",
    "wood-anderson-old",
    "wood-anderson-2800",
]
# All models from Bormann (2012) https://doi.org/10.2312/GFZ.NMSOP-2_DS_3.1
# Page 5

COMPONENT_MAP: dict[Component, ChannelSelector] = {
    "all": ChannelSelectors.All,
    "all-abs": ChannelSelectors.AllAbsolute,
    "horizontal-abs": ChannelSelectors.HorizontalAbs,
    "horizontal-avg": ChannelSelectors.HorizontalAvg,
    "vertical": ChannelSelectors.Vertical,
    "horizontal-separate": ChannelSelectors.Horizontal,
    "north-east-separate": ChannelSelectors.NorthEast,
}

_QUANTITY_MAP: dict[MaxAmplitudeQuantity, MeasurementUnit] = {
    "counts": "counts",
    "acceleration": "acceleration",
    "velocity": "velocity",
    "displacement": "displacement",
    "wood-anderson": "displacement",
    "wood-anderson-old": "displacement",
    "wood-anderson-2800": "displacement",
}

# New corrections from NMSOP 3.3
# 10.2312/GFZ.NMSOP-2_IS_3.3, Page 5
WOOD_ANDERSON = PoleZeroResponse(
    poles=[
        -5.49779 + 5.60886j,
        -5.49779 - 5.60886j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

WOOD_ANDERSON_OLD = PoleZeroResponse(
    poles=[
        -6.283 + 4.7124j,
        -6.283 - 4.7124j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

WOOD_ANDERSON_PIDSA = PoleZeroResponse(
    poles=[
        -6.283 + 4.7124j,
        -6.283 - 4.7124j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2800.0,
)


class Range(NamedTuple):
    min: float
    max: float

    def inside(self, value: float) -> bool:
        return self.min <= value <= self.max


class StationLocalMagnitude(NamedTuple):
    station: NSL
    magnitude: float
    error: float
    peak_amp: float
    distance_epi: float
    distance_hypo: float
    snr: float = 0.0


class LocalMagnitudeModel:
    epicentral_range: ClassVar[Range | None] = None
    hypocentral_range: ClassVar[Range | None] = None

    max_amplitude: ClassVar[MaxAmplitudeQuantity] = "wood-anderson"

    component: ClassVar[Component] = "vertical"
    peak_measurement: ClassVar[PeakMeasurement] = "peak-to-peak"

    # Used only for non-Wood-Anderson models
    highpass_freq: ClassVar[float | None] = None
    lowpass_freq: ClassVar[float | None] = None

    author: ClassVar[str] = "Unknown"
    doi: ClassVar[str] = ""

    hypo_depth_only: ClassVar[bool] = False

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

    @property
    def restitution_quantity(self) -> MeasurementUnit:
        return _QUANTITY_MAP[self.max_amplitude]

    def get_amplitude_traces(self, traces: list[Trace]) -> list[Trace]:
        return COMPONENT_MAP[self.component].get_traces(traces)

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
        station_nsl: NSL,
    ) -> float:
        raise NotImplementedError


class WebnetWesternBohemia(LocalMagnitudeModel):
    author = "Horálek et al. (2000)"
    doi = "10.1023/A:1022198406514"

    hypocentral_range = Range(0.0 * KM, 100.0 * KM)
    component = "all-abs"
    max_amplitude = "velocity"

    peak_measurement = "max-amplitude"
    highpass_freq = 1.0

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
        station_nsl: NSL,
    ) -> float:
        return (
            np.log10(amplitude * UM)
            - np.log10(np.pi)
            + 2.1 * np.log10(distance_hypo / KM)
            - 1.7
        )


class WoodAnderson:
    max_amplitude: ClassVar[MaxAmplitudeQuantity] = "wood-anderson"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        """Get the amplitude attenuation for a given distance, the `-log10(A_0)`term."""
        raise NotImplementedError

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
        station_nsl: NSL,
    ) -> float:
        """Get the magnitude from a given amplitude."""
        amplitude = amplitude * MM  # m to mm
        return np.log10(amplitude) + self.get_amp_attenuation(
            distance_hypo / KM, distance_epi / KM
        )


class IaspeiSouthernCalifornia(LocalMagnitudeModel):
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "north-east-separate"
    max_amplitude = "wood-anderson-old"

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
        station_nsl: NSL,
    ) -> float:
        amp = amplitude * M2NM  # m to nm
        return (
            np.log10(amp / WOOD_ANDERSON.constant)
            + 1.11 * np.log10(distance_hypo / KM)
            + 0.00189 * (distance_hypo / KM)
            - 2.09
        )


class SouthernCalifornia(WoodAnderson, LocalMagnitudeModel):
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "north-east-separate"
    max_amplitude = "wood-anderson-old"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class CentralCalifornia(WoodAnderson, LocalMagnitudeModel):
    author = "Bakun and Joyner (1984)"
    doi = "10.1785/BSSA0740051827"

    epicentral_range = Range(0.0 * KM, 400.0 * KM)
    component = "horizontal-avg"
    max_amplitude = "wood-anderson-old"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return np.log10(dist_hypo_km) + 0.00301 * dist_hypo_km + 0.7


class EasternNorthAmerica(WoodAnderson, LocalMagnitudeModel):
    author = "Kim (1998)"

    epicentral_range = Range(100.0 * KM, 800.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * np.log10(dist_epi_km) - 0.22


class Albania(WoodAnderson, LocalMagnitudeModel):
    author = "Muco and Minga (1991)"

    epicentral_range = Range(10.0 * KM, 600.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * np.log10(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(WoodAnderson, LocalMagnitudeModel):
    author = "Stange (2006)"

    hypocentral_range = Range(10.0 * KM, 1000.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(WoodAnderson, LocalMagnitudeModel):
    author = "Greenhalgh and Singh (1986)"

    epicentral_range = Range(40.0 * KM, 700.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * np.log10(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(WoodAnderson, LocalMagnitudeModel):
    author = "Alsaker et al. (1991)"

    hypocentral_range = Range(0.0 * KM, 1500.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * np.log10(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


class IcelandAskja(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.4406 * np.log10(dist_hypo_km / 17) + 0.003 * (dist_hypo_km - 17) + 2


class IcelandBardabunga(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.2534 * np.log10(dist_hypo_km / 17) - 0.0032 * (dist_hypo_km - 17) + 2


class IcelandAskjaBardabungaCombined(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1999 * np.log10(dist_hypo_km / 17) - 0.0016 * (dist_hypo_km - 17) + 2


class IcelandReykjanes(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2022)"

    hypocentral_range = Range(0.0 * KM, 40.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.6902 * np.log10(dist_hypo_km / 17) + 0.0318 * (dist_hypo_km - 17) + 2


class Azores(WoodAnderson, LocalMagnitudeModel):
    author = "Gongora et al. (2004)"

    epicentral_range = Range(10.0 * KM, 800.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.89 * np.log10(dist_epi_km / 100) + 0.00256 * (dist_epi_km - 100) + 3


class ArgentinaVolcanoes(WoodAnderson, LocalMagnitudeModel):
    author = "Montenegro et al. (2021)"

    epicentral_range = Range(0.0 * KM, 100.0 * KM)  # Bounds are not clear
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 2.76 * np.log10(dist_epi_km) - 2.48


class NetherlandsGroningen(WoodAnderson, LocalMagnitudeModel):
    author = "Dost et al. (2018)"

    epicentral_range = Range(0.0 * KM, 80.0 * KM)
    component = "horizontal-avg"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.33 * np.log10(dist_hypo_km) + 0.00139 * dist_hypo_km + 0.424


class CampiFlegrei(WoodAnderson, LocalMagnitudeModel):
    author = "Petrosino et al. (2008)"
    doi = "10.1785/0120070131"

    max_amplitude = "wood-anderson-2800"
    epicentral_range = Range(0.0 * KM, 10.0 * KM)  # The paper says 0.2 - 8 km
    component = "horizontal-avg"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.95 * np.log10(dist_hypo_km) + 0.09 * dist_hypo_km - 0.1


ModelName = Literal[LocalMagnitudeModel.model_names()]


class AttenuationModel(Model):
    a: float
    b: float
    c: float

    def get_amp_attenuation(
        self,
        dist_hypo_km: float,
        dist_epi_km: float,
        distance: Literal["epicentral", "hypocentral"] = "epicentral",
    ) -> float:
        distance = (
            dist_hypo_km
            if (distance or self.distance) == "hypocentral"
            else dist_epi_km
        )
        return self.a * np.log10(distance) + self.b * distance + self.c


class CustomLocalMagnitudeModel(Model):
    max_amplitude: MaxAmplitudeQuantity = "wood-anderson"

    distance: Literal["epicentral", "hypocentral"] = "hypocentral"
    epicentral_range: Range | None = None
    hypocentral_range: Range | None = None

    component: Component = "horizontal-abs"
    peak_measurement: PeakMeasurement = "peak-to-peak"
    hypo_depth_only_legacy: bool = Field(False, exclude=True)

    # Used only for non-Wood-Anderson models
    highpass_freq: float | None = None
    lowpass_freq: float | None = None

    attenuation_models: dict[NSL, AttenuationModel] = Field(default_factory=dict)

    _model: LocalMagnitudeModel = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        _attenuation_models = self.attenuation_models
        if not _attenuation_models:
            raise ValueError("At least one attenuation model must be provided.")

        class CustomModel(LocalMagnitudeModel):
            max_amplitude = self.max_amplitude

            distance = self.distance
            epicentral_range = self.epicentral_range
            hypocentral_range = self.hypocentral_range

            component = self.component
            peak_measurement = self.peak_measurement

            highpass_freq = self.highpass_freq
            lowpass_freq = self.lowpass_freq

            hypo_depth_only_legacy = self.hypo_depth_only_legacy

            def get_magnitude(
                self,
                amplitude: float,
                distance_hypo: float,
                distance_epi: float,
                station_nsl: NSL,
            ) -> float:
                if self.max_amplitude in ("wood-anderson", "wood-anderson-old"):
                    amplitude = amplitude * MM

                for nsl, model in _attenuation_models.items():
                    if nsl.match(station_nsl):
                        return np.log10(amplitude) + model.get_amp_attenuation(
                            distance_hypo / KM,
                            distance_epi / KM,
                            self.distance,
                        )
                else:
                    raise ValueError(f"No attenuation model for {station_nsl.pretty}")

        self._model = CustomModel()
        return self

    def get_model(self) -> LocalMagnitudeModel:
        return self._model
