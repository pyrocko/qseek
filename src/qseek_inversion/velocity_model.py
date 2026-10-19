from __future__ import annotations

import numpy as np

from qseek.models.layered_model import LayeredModel


class LayeredModelInversion(LayeredModel):
    def perturb_velocities(
        self,
        noise: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Perturb the velocities of each layer by a random relative factor.

        The P velocity of each layer is scaled by a factor drawn uniformly from
        `[1 - noise, 1 + noise]`. The S velocity follows, keeping the layer's Vp/Vs
        ratio.

        Args:
            noise (float): Maximum relative perturbation, e.g. 0.1 for ±10%.
            rng (np.random.Generator | None, optional): Random number generator.
                Defaults to a new, unseeded generator.

        Raises:
            ValueError: If noise is not in (0, 1).
        """
        if not 0.0 < noise < 1.0:
            raise ValueError("noise must be between 0 and 1")
        rng = rng or np.random.default_rng()
        for layer in self.layers:
            vpvs = layer.vpvs
            layer.vp *= 1.0 + rng.uniform(-noise, noise)
            layer.vs = layer.vp / vpvs

    @classmethod
    def from_layered_model(cls, model: LayeredModel) -> LayeredModelInversion:
        """Create an independent copy of a layered model.

        Args:
            model (LayeredModel): Model to copy. Perturbing the copy leaves it
                unchanged.

        Returns:
            LayeredModelInversion: The copy.
        """
        return cls.model_validate(model.model_dump())
