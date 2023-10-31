from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

import numpy as np

from lassie.plot.base import BasePlot, LassieFigure

HOUR = 3600
DAY = 24 * HOUR


DetectionAttribute = Literal["semblance", "magnitude"]


class DetectionsDistribution(BasePlot):
    attribute: ClassVar[DetectionAttribute] = "semblance"

    def get_figure(self) -> LassieFigure:
        return self.create_figure(attribute=self.attribute)

    def create_figure(
        self,
        attribute: DetectionAttribute = "semblance",
    ) -> LassieFigure:
        figure = self.new_figure("event-distribution-{attribute}.png")
        axes = figure.get_axes()

        detections = self.detections

        values = [getattr(detection, attribute) for detection in detections]
        times = [
            detection.time.replace(tzinfo=None)  # Stupid fix for matplotlib bug
            for detection in detections
        ]

        axes.scatter(times, values, cmap="viridis_r", c=values, s=3, alpha=0.5)
        axes.set_ylabel(attribute.capitalize())
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
        from_timestamps = np.vectorize(
            lambda t: datetime.fromtimestamp(t, tz=timezone.utc)
        )
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
        return figure
