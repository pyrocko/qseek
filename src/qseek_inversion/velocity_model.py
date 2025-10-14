from __future__ import annotations

import numpy as np

from qseek.models.layered_model import LayeredModel


class LayeredModelInversion(LayeredModel):
    def perturb_velocities(self, noise: float) -> None:
        """Perturb the velocities of the model by a given factor.

        Args:
            noise (float): Maximum perturbation factor (e.g., 0.1 for ±10%).

        Raises:
            ValueError: If noise is not positive.
        """
        if noise <= 0:
            raise ValueError("Noise must be a positive value.")
        for layer in self.layers:
            pert = np.random.uniform(noise, noise)
            layer.vp *= 1 + pert
            layer.vs *= 1 + pert

    @classmethod
    def from_layered_model(cls, model: LayeredModel) -> LayeredModelInversion:
        return cls(layers=model.layers)
