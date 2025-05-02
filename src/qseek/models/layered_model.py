from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from scipy.interpolate import interp1d

from qseek.tracers.utils import EarthModel

logger = logging.getLogger(__name__)


class Layer(BaseModel):
    top_depth: float = Field()
    vp: float = Field(ge=0.0)
    vs: float = Field(ge=0.0)

    @property
    def vpvs(self) -> float:
        """Calculate the Vp/Vs ratio."""
        return self.vp / self.vs


class LayeredModel(BaseModel):
    layers: list[Layer] = Field(min_length=1)

    _interpolator_vp: interp1d = PrivateAttr()
    _interpolator_vs: interp1d = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self.layers = sorted(self.layers, key=lambda x: x.top_depth)

        mid_points = [
            (layer.top_depth + self.layers[i + 1].top_depth) / 2
            for i, layer in enumerate(self.layers[:-1])
        ]
        vp = [layer.vp for layer in self.layers[:-1]]
        vs = [layer.vs for layer in self.layers[:-1]]
        self._interpolator_vp = interp1d(
            mid_points,
            vp,
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",  # type: ignore
        )
        self._interpolator_vs = interp1d(
            mid_points,
            vs,
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",  # type: ignore
        )

    def vp_interpolator(self, depth: float | np.ndarray) -> np.ndarray:
        """Get the Vp value at a given depth."""
        return self._interpolator_vp(depth)

    def vs_interpolator(self, depth: float | np.ndarray) -> float | np.ndarray:
        """Get the Vs value at a given depth."""
        return self._interpolator_vs(depth)

    @classmethod
    def from_earth_model(cls, earth_model: EarthModel) -> LayeredModel:
        """Create a LayeredModel from an EarthModel."""
        layers = []
        for layer in earth_model.layered_model.layers():
            layers.append(
                Layer(
                    top_depth=layer.ztop,
                    vp=layer.m.vp,
                    vs=layer.m.vs,
                )
            )
        return cls(layers=layers)
