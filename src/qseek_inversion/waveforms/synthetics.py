from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveInt, PrivateAttr

from qseek.images.base import WaveformImage
from qseek.images.images import WaveformImages
from qseek.search import Search
from qseek.synthetics.synthetics import SyntheticEvent, SyntheticEventCatalog
from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.tracers import RayTracerType
from qseek.utils import PhaseDescription
from qseek_inversion.waveforms.base import WaveformSelection

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.station import StationInventory


class SyntheticEvents(WaveformSelection):
    """Select synthetic waveforms for a list of synthetic events."""

    waveforms: Literal["SyntheticEvents"] = "SyntheticEvents"

    import_rundir: Path = Field(
        default=...,
        description="Import rundir to get search configuration from.",
    )

    ray_tracer: RayTracerType = Field(
        default_factory=FastMarchingTracer.model_construct,
        description="Ray tracer to use for travel time calculations.",
    )
    n_events: PositiveInt = Field(
        default=10,
        description="Number of synthetic events to generate waveforms for.",
    )
    inter_event_time: timedelta = Field(
        default=timedelta(seconds=10),
        description="Time difference between synthetic events.",
    )
    sampling_rate: float = Field(
        default=100.0,
        description="Sampling rate for synthetic waveforms.",
    )
    sigma_samples: PositiveInt = Field(
        default=10,
        description="Sigma samples for Gaussian source time function.",
    )
    batch_length: timedelta = Field(
        default=timedelta(minutes=10),
        description="Number of events to include in each waveform batch.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for synthetic event generation.",
    )

    _start_time: datetime = PrivateAttr(
        default=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    _catalog: SyntheticEventCatalog = PrivateAttr()

    async def prepare(self, rundir: Path | None = None) -> None:
        search = Search.from_config(self.import_rundir)
        octree = search.octree

        search.stations.sanitize_stations()
        search.stations.filter_stations(search.data_provider.available_nsls())
        self.stations = search.stations

        await self.ray_tracer.prepare(search.octree, search.stations)

        rnd = np.random.default_rng(self.random_seed)

        north_shifts = rnd.uniform(
            low=octree.north_bounds[0] + octree.root_node_size,
            high=octree.north_bounds[1] - octree.root_node_size,
            size=self.n_events,
        )
        east_shifts = rnd.uniform(
            low=octree.east_bounds[0] + octree.root_node_size,
            high=octree.east_bounds[1] - octree.root_node_size,
            size=self.n_events,
        )
        depths = rnd.uniform(
            low=octree.depth_bounds[0] + octree.root_node_size,
            high=octree.depth_bounds[1] - octree.root_node_size,
            size=self.n_events,
        )

        # for coord in zip(north_shifts, east_shifts, depths, strict=False):
        #     print(coord)

        self._catalog = SyntheticEventCatalog()

        for i in range(self.n_events):
            self._catalog.add_event(
                SyntheticEvent(
                    origin_time=self._start_time + i * self.inter_event_time,
                    lat=octree.location.lat,
                    lon=octree.location.lon,
                    north_shift=north_shifts[i],
                    east_shift=east_shifts[i],
                    depth=depths[i] + octree.location.depth,
                    elevation=octree.location.elevation,
                ),
                stations=search.stations,
                ray_tracer=self.ray_tracer,
            )

        if rundir is not None:
            self._catalog.export_catalog(rundir / "synthetic_catalog")

    def get_traces(
        self,
        phase: PhaseDescription,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Trace]:
        return self._catalog.get_traces(
            phase=phase, start_time=start_time, end_time=end_time
        )

    def get_stations(self) -> StationInventory:
        return self.stations

    async def get_images(self, window_padding: timedelta) -> list[WaveformImages]:
        catalog = self._catalog

        phases = catalog.get_available_phases()
        images: list[WaveformImages] = []

        start_catalog, end_catalog = catalog.get_time_span()
        start_catalog -= window_padding
        end_catalog += window_padding

        current_start = start_catalog
        while current_start < end_catalog:
            current_end = current_start + self.batch_length
            current_end = min(current_end, end_catalog)
            waveform_images = WaveformImages(
                start_time=current_start,
                end_time=current_end,
            )
            for phase in phases:
                traces = catalog.get_traces(
                    phase=phase,
                    start_time=current_start - window_padding,
                    end_time=current_end + window_padding,
                    sampling_rate=self.sampling_rate,
                    sigma_samples=self.sigma_samples,
                )
                if not traces:
                    continue

                img = WaveformImage(
                    image_function="SyntheticArrival",
                    traces=traces,
                    phase=phase,
                    weight=1.0,
                    detection_half_width=self.sigma_samples / self.sampling_rate,
                )
                img.set_stations(self.stations)
                waveform_images.add_image(img)

            images.append(waveform_images)
            current_start = current_end
        return images
