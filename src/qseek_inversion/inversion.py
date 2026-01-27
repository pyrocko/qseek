from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, DirectoryPath, Field, PrivateAttr

from qseek.models.detection import EventDetection
from qseek.search import OctreeSearch, Search
from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.tracers import RayTracers
from qseek.utils import datetime_now
from qseek_inversion.velocity_model import LayeredModelInversion
from qseek_inversion.waveforms import WaveformSelectionType
from qseek_inversion.waveforms.synthetics import SyntheticEvents

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.images.images import WaveformImages

logger = logging.getLogger(__name__)


class InversionLocationResult(BaseModel):
    cumulative_semblance: float = Field(
        default=0.0,
        description="Cumulative semblance of all events.",
    )
    event: list[EventDetection] = Field(
        default_factory=list,
        description="List of detected events.",
    )

    _trace: Trace = PrivateAttr()

    def add_events(self, events: list[EventDetection]) -> None:
        self.event.extend(events)
        self.cumulative_semblance += sum(event.semblance for event in events)

    def set_semblance_trace(self, trace: Trace) -> None:
        self._trace = trace


class InversionLayered1D(BaseModel):
    import_rundir: DirectoryPath = Field(
        default_factory=Path.cwd,
        description="Path to the qseek run directory.",
    )

    event_selection: WaveformSelectionType = Field(
        default_factory=SyntheticEvents.model_construct,
        description="Settings for event and waveform selection.",
    )

    _search: Search = PrivateAttr()
    _octree_search: OctreeSearch = PrivateAttr()
    _images: list[WaveformImages] = PrivateAttr(default_factory=list)

    async def prepare(self) -> None:
        search = Search.from_config(self.import_rundir)

        await self.event_selection.prepare(rundir=self.import_rundir)
        # await search.prepare()
        self._search = search
        if search.station_weights:
            search.station_weights.prepare(
                self.event_selection.stations,
                search.octree,
            )

        await search.ray_tracers.prepare(search.octree, self.event_selection.stations)

        # window_padding = await search.get_window_padding()

        self._images = await self.event_selection.get_images(timedelta(seconds=30))

        self._octree_search = OctreeSearch(
            ray_tracers=search.ray_tracers,
            window_padding=timedelta(seconds=10),
            detection_threshold=search.detection_threshold,
            detection_blinding=search.detection_blinding,
            pick_confidence_threshold=search.pick_confidence_threshold,
            station_corrections=search.station_corrections,
            distance_weights=search.station_weights,
            ignore_boundary=search.ignore_boundary,
            ignore_boundary_width=search.ignore_boundary_width,
            node_interpolation=False,
            attach_arrivals=False,
            neighbor_search=True,
            semblance_density_search=True,
        )

    def get_start_model(self) -> LayeredModelInversion:
        ray_tracer: FastMarchingTracer = self._search.ray_tracers.root[0]
        return LayeredModelInversion.from_layered_model(ray_tracer.get_layered_model())

    async def test_velocity_model(
        self,
        model: LayeredModelInversion,
    ) -> InversionLocationResult:
        search = self._search
        octree_search = self._octree_search

        ray_tracer = FastMarchingTracer(
            velocity_model=None,
            nthreads=0,
            interpolation_method="linear",
            implementation="pyrocko",
        )
        ray_tracer.set_layered_model(model)
        await ray_tracer.prepare(search.octree.reset(), self.event_selection.stations)

        octree_search.set_ray_tracers(RayTracers([ray_tracer]))

        result = InversionLocationResult()
        start = datetime_now()

        for waveform_image in self._images:
            events, trace = await octree_search.search(
                images=waveform_image,
                octree=search.octree.reset(),
                n_threads_parstack=0,
                n_threads_argmax=0,
            )
            result.add_events(events)
            result.set_semblance_trace(trace)

        logger.info(
            "Located %d events in %s, cumulative semblance: %.2f",
            len(result.event),
            datetime_now() - start,
            result.cumulative_semblance,
        )

        return result
