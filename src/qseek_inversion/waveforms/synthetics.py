from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveInt, PrivateAttr

from qseek.images.base import WaveformImage, WaveformImages
from qseek.synthetics.synthetics import SyntheticEvent, SyntheticEventCatalog
from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.tracers import RayTracerType
from qseek.utils import PhaseDescription
from qseek_inversion.waveforms.base import WaveformSelection

if TYPE_CHECKING:
    from pathlib import Path

    from pyrocko.trace import Trace

    from qseek.search import Search


class SyntheticEvents(WaveformSelection):
    """Synthetic image functions of random events inside the octree.

    The arrivals are modelled with `ray_tracer`, the true velocity model of the
    synthetic test. Each arrival is a Gaussian peak in the image function.
    """

    waveforms: Literal["SyntheticEvents"] = "SyntheticEvents"

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
        description="Length of each batch of synthetic images.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for synthetic event generation.",
    )

    _start_time: datetime = PrivateAttr(
        default=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    _catalog: SyntheticEventCatalog = PrivateAttr()

    async def prepare(
        self,
        search: Search,
        import_rundir: Path,
        rundir: Path | None = None,
    ) -> None:
        await super().prepare(search, import_rundir, rundir)
        octree = search.octree
        stations = self.get_stations()

        await self.ray_tracer.prepare(octree, stations)

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

        self._catalog = SyntheticEventCatalog()

        # The same frame as the octree nodes, see Node.as_location
        reference = octree.location
        for i in range(self.n_events):
            self._catalog.add_event(
                SyntheticEvent(
                    origin_time=self._start_time + i * self.inter_event_time,
                    lat=reference.lat,
                    lon=reference.lon,
                    elevation=reference.elevation,
                    north_shift=reference.north_shift + north_shifts[i],
                    east_shift=reference.east_shift + east_shifts[i],
                    depth=reference.depth + depths[i],
                ),
                stations=stations,
                ray_tracer=self.ray_tracer,
            )

        if rundir is not None:
            self._catalog.export_catalog(rundir / "synthetic_catalog")

    def get_catalog(self) -> SyntheticEventCatalog:
        """The synthetic events."""
        return self._catalog

    def get_traces(
        self,
        phase: PhaseDescription,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Trace]:
        return self._catalog.get_traces(
            phase=phase,
            start_time=start_time,
            end_time=end_time,
            sampling_rate=self.sampling_rate,
            sigma_samples=self.sigma_samples,
        )

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
                traces = self.get_traces(
                    phase=phase,
                    start_time=current_start - window_padding,
                    end_time=current_end + window_padding,
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
                img.set_stations(self.get_stations())
                waveform_images.add_image(img)

            images.append(waveform_images)
            current_start = current_end
        return images
