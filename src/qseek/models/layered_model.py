from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np
from pydantic import BaseModel, Field
from pyrocko.cake import GradientLayer as PyrockoGradientLayer
from pyrocko.cake import Layer as PyrockoLayer

from qseek.tracers.utils import LayeredEarthModel1D

logger = logging.getLogger(__name__)


class Layer(BaseModel):
    top_depth: float = Field(
        description="Top depth of the layer in m.",
    )
    vp: float = Field(
        ge=0.0,
        description="P wave velocity in m/s",
    )
    vs: float = Field(
        ge=0.0,
        description="S wave velocity in m/s",
    )

    gradient_vp: float = Field(
        default=0.0,
        description="Velocity gradient in m/s/m.",
    )
    gradient_vs: float = Field(
        default=0.0,
        description="Velocity gradient in m/s/m.",
    )

    def _interpolate_vp(self, depth: np.ndarray) -> np.ndarray:
        return self.vp + self.gradient_vp * (depth - self.top_depth)

    def _interpolate_vs(self, depth: np.ndarray) -> np.ndarray:
        return self.vs + self.gradient_vs * (depth - self.top_depth)

    @property
    def vpvs(self) -> float:
        """Calculate the Vp/Vs ratio."""
        return self.vp / self.vs

    @classmethod
    def from_pyrocko(cls, layer: PyrockoLayer | PyrockoGradientLayer) -> Layer:
        """Create a Layer from a pyrocko.cake.Layer."""
        if isinstance(layer, PyrockoGradientLayer):
            thickness = layer.zbot - layer.ztop
            gradient_vp = (layer.mbot.vp - layer.mtop.vp) / thickness
            gradient_vs = (layer.mbot.vs - layer.mtop.vs) / thickness
            return cls(
                top_depth=layer.ztop,
                vp=layer.mtop.vp,
                vs=layer.mtop.vs,
                gradient_vp=gradient_vp,
                gradient_vs=gradient_vs,
            )
        if isinstance(layer, PyrockoLayer):
            return cls(
                top_depth=layer.ztop,
                vp=layer.m.vp,
                vs=layer.m.vs,
            )
        raise ValueError(
            f"Layer type {type(layer)} is not supported. "
            "Use pyrocko.cake.Layer or pyrocko.cake.GradientLayer."
        )


class LayeredModel(BaseModel):
    layers: list[Layer] = Field(
        min_length=1,
        description="List of velocity layers.",
    )

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def model_post_init(self, context: Any) -> None:
        self.layers = sorted(self.layers, key=lambda x: x.top_depth)

    def _get_layer(self, depth: float) -> Layer:
        idx = int(self._get_layer_index(np.asarray([depth]))[0])
        return self.layers[idx]

    def _get_layer_index(self, depths: np.ndarray) -> np.ndarray:
        """Get the layer index for each depth."""
        indices = np.searchsorted(
            [layer.top_depth for layer in self.layers],
            depths,
            side="right",
        )
        return np.clip(indices - 1, 0, self.n_layers)

    def _interpolator(
        self, depths: np.ndarray, func: Callable[[Layer, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Apply a function to each layer at the given depths.

        Args:
            depths (np.ndarray): Depth or array of depths in meters.
            func (Callable): Function to apply to each layer.

        Returns:
            np.ndarray: Array of Vp values at the given depths.
        """
        layer_indices = self._get_layer_index(depths)
        velocities = np.zeros_like(depths, dtype=float)
        cum_mask = np.zeros_like(depths, dtype=bool)

        for layer_idx in np.unique(layer_indices):
            layer = self.layers[int(layer_idx)]
            mask = layer_indices == layer_idx
            cum_mask |= mask
            velocities[mask] = func(layer, depths[mask])

        if not np.all(cum_mask):
            raise ValueError("Some depths are out of the model range.")

        return velocities

    def vp_interpolator(self, depths: np.ndarray) -> np.ndarray:
        """Get the Vp value at a given depth.

        Args:
            depths (float | np.ndarray): Depth or array of depths in meters.

        Returns:
            np.ndarray: Array of Vp values at the given depths.
        """
        return self._interpolator(
            depths, lambda layer, depths: layer._interpolate_vp(depths)
        )

    def vs_interpolator(self, depths: np.ndarray) -> np.ndarray:
        """Get the Vs value at a given depth.

        Args:
            depths (float | np.ndarray): Depth or array of depths in meters.

        Returns:
            np.ndarray: Array of Vp values at the given depths.
        """
        return self._interpolator(
            depths, lambda layer, depths: layer._interpolate_vs(depths)
        )

    def plot(
        self,
        depth_range: tuple[float, float],
        samples: int = 100,
        export: Path | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()

        depths = np.linspace(*depth_range, samples)

        vp = self.vp_interpolator(depths)
        vs = self.vs_interpolator(depths)

        ax.plot(vp, depths, label="Vp")
        ax.plot(vs, depths, label="Vs")

        for layer in self.layers:
            ax.axhline(layer.top_depth, color="k", linestyle="--", alpha=0.5)

        ax.set_ylim(*depth_range)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.3)

        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.1f}")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.1f}")
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")

        if export is not None:
            fig.savefig(export, dpi=300)
            plt.close()
        else:
            plt.show()

    @classmethod
    def from_earth_model(cls, earth_model: LayeredEarthModel1D) -> Self:
        """Create a LayeredModel from an EarthModel."""
        layers = []
        for layer in earth_model.layered_model.layers():
            layers.append(Layer.from_pyrocko(layer))
        return cls(layers=layers)
