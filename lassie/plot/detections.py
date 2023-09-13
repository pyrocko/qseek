from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from .utils import with_default_axes

if TYPE_CHECKING:
    from lassie.models.detection import EventDetections

HOUR = 3600
DAY = 24 * HOUR


@with_default_axes
def plot_detections(
    detections: EventDetections,
    axes: plt.Axes | None = None,
    filename: Path | None = None,
) -> None:
    axes = cast(plt.Axes, axes)  # injected by wrapper

    semblances = [detection.semblance for detection in detections]
    times = [
        detection.time.replace(tzinfo=None)  # Stupid fix for matplotlib bug
        for detection in detections
    ]

    axes.scatter(times, semblances, cmap="viridis_r", c=semblances, s=3, alpha=0.5)
    axes.set_ylabel("Detection Semblance")
    axes.grid(axis="x", alpha=0.3)
    # axes.figure.autofmt_xdate()

    cum_axes = axes.twinx()

    cummulative_detections = np.cumsum(np.ones(detections.n_detections))
    cum_axes.plot(
        times,
        cummulative_detections,
        color="black",
        alpha=0.8,
        label="Cumulative Detections",
    )
    cum_axes.set_ylabel("# Detections")

    to_timestamps = np.vectorize(lambda d: d.timestamp())
    from_timestamps = np.vectorize(lambda t: datetime.fromtimestamp(t, tz=timezone.utc))
    detection_time_span = times[-1] - times[0]
    daily_rate, edges = np.histogram(
        to_timestamps(times),
        bins=detection_time_span.days,
    )

    cum_axes.stairs(
        daily_rate,
        from_timestamps(edges),
        color="gray",
        fill=True,
        alpha=0.5,
        label="Daily Detections",
    )
    cum_axes.legend(loc="upper left", fontsize="small")
