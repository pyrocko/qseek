from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import AwareDatetime, Field, PositiveInt, PrivateAttr
from pyrocko.trace import Trace

from qseek.images.base import ImageFunction, PhaseName, WaveformImage
from qseek.models.station import Station
from qseek.synthetics.synthetics import SyntheticEvent
from qseek.tracers.base import RayTracer
from qseek.tracers.fast_marching import FastMarchingTracer

if TYPE_CHECKING:
    from qseek.models.station import Stations
    from qseek.octree import Octree
    from qseek.utils import PhaseDescription
    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)


def _gaussian_function(sigma: int = 10):
    length = sigma * 10
    onset = length // 2
    x = np.arange(length)
    return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


class SyntheticImage(WaveformImage): ...


class Synthetic(ImageFunction):
    image: Literal["synthetic"] = "synthetic"

    start_time: AwareDatetime = Field(
        default=datetime(2025, 1, 1, tzinfo=timezone.utc),
        description="Start time for synthetic traces.",
    )
    end_time: AwareDatetime = Field(
        default=datetime(2025, 1, 2, tzinfo=timezone.utc),
        description="End time for synthetic traces.",
    )
    ray_tracer: RayTracer = Field(
        default_factory=FastMarchingTracer.model_construct,
        description="Ray tracer to use for travel time calculations.",
    )

    max_events: PositiveInt = Field(
        default=100,
        description="Number of synthetic events to generate.",
    )
    min_time_between_events: timedelta = Field(
        default=timedelta(minutes=1),
        description="Minimum time between synthetic events.",
    )
    sampling_rate: PositiveInt = Field(
        default=100,
        description="Sampling rate for synthetic traces in Hz.",
    )
    sigma_pick_samples: PositiveInt = Field(
        default=10,
        description="Standard deviation in samples for Gaussian pick simulation.",
    )

    phase_map: dict[PhaseName, str] = Field(
        default={
            "P": "cake:P",
            "S": "cake:S",
        },
        description="Phase mapping from SeisBench PhaseNet to "
        "Qseek travel time phases.",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed for synthetic event generation.",
    )

    _stations: Stations = PrivateAttr()
    _octree: Octree = PrivateAttr()

    _events: list[SyntheticEvent] = PrivateAttr(default_factory=list)

    async def prepare(
        self,
        stations: Stations,
        octree: Octree,
        rundir: Path | None = None,
    ) -> None:
        logger.debug("preparing synthetic image function")
        self._stations = stations
        self._octree = octree

        await self.ray_tracer.prepare(octree, stations)
        self._events = self.create_random_events()

        if rundir is not None:
            export_folder = rundir / "synthetic_events"
            export_folder.mkdir(parents=True, exist_ok=True)
            self.export_csv(export_folder / "synthetic_events.csv")

    def create_random_events(self) -> list[SyntheticEvent]:
        octree = self._octree
        rnd = np.random.default_rng(self.random_seed)

        north_shifts = rnd.uniform(
            low=octree.north_bounds[0] - octree.root_node_size,
            high=octree.north_bounds[1] - octree.root_node_size,
            size=self.max_events,
        )
        east_shifts = rnd.uniform(
            low=octree.east_bounds[0] - octree.root_node_size,
            high=octree.east_bounds[1] - octree.root_node_size,
            size=self.max_events,
        )
        depths = rnd.uniform(
            low=octree.depth_bounds[0] - octree.root_node_size,
            high=octree.depth_bounds[1] - octree.root_node_size,
            size=self.max_events,
        )
        times = rnd.uniform(
            low=self.start_time.timestamp(),
            high=self.end_time.timestamp(),
            size=self.max_events,
        )
        times.sort()
        time_between = np.diff(
            times,
            append=times[-1] + self.min_time_between_events.total_seconds(),
        )
        mask = time_between < self.min_time_between_events.total_seconds()

        events = []
        for i in range(self.max_events):
            if mask[i]:
                continue
            event = SyntheticEvent(
                lat=octree.location.lat,
                lon=octree.location.lon,
                north_shift=north_shifts[i],
                east_shift=east_shifts[i],
                depth=depths[i] + octree.location.depth,
                elevation=octree.location.elevation,
                origin_time=datetime.fromtimestamp(times[i], tz=timezone.utc),
            )
            events.append(event)

        logger.info("created %d synthetic events", len(events))
        return events

    def export_csv(self, filename: Path) -> None:
        with filename.open("w") as f:
            f.write("origin_time,lat,lon,depth,magnitude,WKT_geom\n")
            for event in self._events:
                f.write(event.as_csv() + "\n")
        logger.info("exported %d synthetic events to %s", len(self._events), filename)

    async def process_traces(self, batch: WaveformBatch) -> list[SyntheticImage]:
        image_functions = []

        for phase in self.ray_tracer.get_available_phases():
            traces = []
            for station in self._stations:
                trace = (
                    await self.get_trace(
                        station=station,
                        phase=phase,
                        start_time=batch.start_time,
                        end_time=batch.end_time,
                        sampling_rate=self.sampling_rate,
                    ),
                )
                traces.append(trace)

            image = SyntheticImage(
                traces=traces,
                image_function=self,
                phase=phase.split(":")[-1],
                weight=1.0,
                detection_half_width=self.sigma_pick_samples / self.sampling_rate,
                stations=self._stations,
            )
            image_functions.append(image)
        return image_functions

    async def get_trace(
        self,
        station: Station,
        phase: PhaseDescription,
        start_time: datetime,
        end_time: datetime,
        sampling_rate: float,
    ) -> Trace:
        n_samples = int((end_time - start_time).total_seconds() * sampling_rate)
        data = np.zeros(n_samples, dtype=np.float32)

        selected_events = [
            ev for ev in self._events if start_time <= ev.origin_time <= end_time
        ]

        for event in selected_events:
            travel_time = self.ray_tracer.get_travel_time_location(
                phase=phase,
                source=event,
                receiver=station,
            )
            arrival_time = event.origin_time + timedelta(seconds=travel_time)
            if not (start_time <= arrival_time <= end_time):
                continue
            sample_idx = int(
                (arrival_time - start_time).total_seconds() * sampling_rate
            )
            if 0 <= sample_idx < n_samples:
                data[sample_idx] = 1.0  # Simple spike for arrival

        data = np.convolve(
            data,
            _gaussian_function(sigma=self.sigma_pick_samples),
            mode="same",
        )

        return Trace(
            network=station.network,
            station=station.station,
            location=station.location,
            channel=phase,
            ydata=data,
            tmin=start_time.timestamp(),
            deltat=1.0 / sampling_rate,
        )

    def get_blinding(self, sampling_rate: float) -> timedelta:
        return timedelta(seconds=0)

    def get_provided_phases(self) -> tuple[PhaseDescription, ...]:
        return tuple(self.phase_map.values())
