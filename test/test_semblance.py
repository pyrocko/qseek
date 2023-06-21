from datetime import datetime, timezone

import numpy as np

from lassie.models.semblance import Semblance


def test_semblance_reset() -> None:
    """Test that the semblance is reset."""
    semblance = Semblance(
        n_nodes=10,
        n_samples=100,
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        sampling_rate=1,
        padding_samples=10,
    )
    semblance.semblance_unpadded[:, :] = 1.0
    assert np.all(semblance.semblance_unpadded == 0.0)
