from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, DirectoryPath, Field, PositiveInt, PrivateAttr

from qseek.models.detection import EventDetection
from qseek.models.layered_model import Layer
from qseek.search import OctreeSearch, Search
from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.tracers import RayTracers
from qseek.utils import datetime_now
from qseek_inversion.velocity_model import LayeredModelInversion
from qseek_inversion.waveforms import WaveformSelectionType
from qseek_inversion.waveforms.synthetics import SyntheticEvents

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.images.base import WaveformImages

logger = logging.getLogger(__name__)

# Slower test models shift the arrivals further than the start model
PADDING_FACTOR = 1.5


class InversionLocationResult(BaseModel):
    cumulative_semblance: float = Field(
        default=0.0,
        description="Cumulative semblance of all events.",
    )
    event: list[EventDetection] = Field(
        default_factory=list,
        description="List of detected events.",
    )

    _traces: list[Trace] = PrivateAttr(default_factory=list)

    def add_events(self, events: list[EventDetection]) -> None:
        self.event.extend(events)
        self.cumulative_semblance += sum(event.semblance for event in events)

    def add_semblance_trace(self, trace: Trace) -> None:
        self._traces.append(trace)

    def get_semblance_traces(self) -> list[Trace]:
        """Semblance traces of all image batches."""
        return self._traces


class InversionIteration(BaseModel):
    """One trial model of the inversion, a line of `iterations.jsonl`."""

    iteration: int = Field(
        description="Number of the trial, 0 is the start model.",
    )
    cumulative_semblance: float = Field(
        description="Cumulative semblance of the located events.",
    )
    n_events: int = Field(
        description="Number of located events.",
    )
    accepted: bool = Field(
        description="Whether the trial model became the current model.",
    )
    layers: list[Layer] = Field(
        description="Layers of the trial model.",
    )


class InversionSummary(BaseModel):
    """Result of an inversion, written to `results.json`."""

    n_iterations: int = Field(
        description="Number of trial models after the start model.",
    )
    n_accepted: int = Field(
        description="Number of accepted trial models.",
    )
    n_selected_events: int | None = Field(
        default=None,
        description="Number of events selected for the inversion.",
    )
    start_semblance: float = Field(
        description="Cumulative semblance of the start model.",
    )
    best_semblance: float = Field(
        description="Cumulative semblance of the best model.",
    )
    best_iteration: int = Field(
        description="Iteration of the best model.",
    )
    started: datetime = Field(
        description="Start time of the inversion.",
    )
    duration: timedelta = Field(
        description="Duration of the inversion.",
    )


class InversionLayered1D(BaseModel):
    """Velocity model of 1D layers that focuses the image functions best.

    Each trial perturbs the P velocities of the current model by a random relative
    factor and keeps the Vp/Vs ratios. A trial that raises the cumulative semblance
    of the located events becomes the current model (hill climbing).
    """

    import_rundir: DirectoryPath = Field(
        default=Path("qseek-rundir"),
        description="Path to the qseek run directory.",
    )

    event_selection: WaveformSelectionType = Field(
        default_factory=SyntheticEvents,
        description="Settings for event and waveform selection.",
    )
    n_iterations: PositiveInt = Field(
        default=100,
        description="Number of trial models.",
    )
    perturbation: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Maximum relative perturbation of the P velocity of each layer "
        "per trial, e.g. 0.05 for ±5%.",
    )
    random_seed: int = Field(
        default=0,
        description="Seed of the random perturbations.",
    )

    _search: Search = PrivateAttr()
    _octree_search: OctreeSearch = PrivateAttr()
    _start_tracer: FastMarchingTracer = PrivateAttr()
    _images: list[WaveformImages] = PrivateAttr(default_factory=list)
    _window_padding: timedelta = PrivateAttr()

    async def prepare(self, rundir: Path | None = None) -> None:
        """Load the imported run and select the waveforms.

        Args:
            rundir (Path | None, optional): Directory to export the selected events
                and waveforms to. Defaults to None.
        """
        search = Search.from_config(self.import_rundir)
        self._search = search
        self._start_tracer = self._get_fast_marching_tracer(search.ray_tracers)

        # The stations of a run directory are part of its search.json
        if not search.stations.stations:
            search.stations.prepare(search.octree.location)
        search.stations.sanitize_stations()

        await self.event_selection.prepare(
            search,
            import_rundir=self.import_rundir,
            rundir=rundir,
        )
        stations = self.event_selection.get_stations()

        if search.station_weights:
            search.station_weights.prepare(stations, search.octree)
        if search.station_corrections:
            logger.warning(
                "ignoring station corrections, "
                "they absorb the errors of the velocity model"
            )

        await search.ray_tracers.prepare(search.octree, stations)

        shift_min, shift_max = await search.ray_tracers.get_travel_time_span(
            search.octree.leaf_nodes, list(stations)
        )
        self._window_padding = (
            shift_max - shift_min
        ) * PADDING_FACTOR + self.event_selection.get_blinding()
        logger.info("window padding length is %s", self._window_padding)

        self._images = await self.event_selection.get_images(self._window_padding)

        self._octree_search = OctreeSearch(
            ray_tracers=search.ray_tracers,
            window_padding=self._window_padding,
            trigger=search.trigger,
            station_weights=search.station_weights,
            ignore_boundary=search.ignore_boundary,
            ignore_boundary_width=search.ignore_boundary_width,
            node_interpolation=False,
            attach_arrivals=False,
            neighbor_search=True,
            semblance_density_search=True,
        )

    @staticmethod
    def _get_fast_marching_tracer(ray_tracers: RayTracers) -> FastMarchingTracer:
        for tracer in ray_tracers:
            if isinstance(tracer, FastMarchingTracer):
                return tracer
        raise TypeError("the inversion needs a FastMarchingTracer as ray tracer")

    def get_start_model(self) -> LayeredModelInversion:
        """A copy of the velocity model of the imported run."""
        return LayeredModelInversion.from_layered_model(
            self._start_tracer.get_layered_model()
        )

    async def test_velocity_model(
        self,
        model: LayeredModelInversion,
    ) -> InversionLocationResult:
        """Locate the selected events in a velocity model.

        Args:
            model (LayeredModelInversion): The velocity model to test.

        Returns:
            InversionLocationResult: The detections and their cumulative semblance.
        """
        search = self._search
        octree_search = self._octree_search
        start_tracer = self._start_tracer

        ray_tracer = FastMarchingTracer(
            velocity_model=None,
            phases=start_tracer.phases,
            interpolation_method=start_tracer.interpolation_method,
            nthreads=start_tracer.nthreads,
        )
        ray_tracer.set_layered_model(model)
        await ray_tracer.prepare(
            search.octree.reset(), self.event_selection.get_stations()
        )

        # Other tracers, e.g. for further phases, stay as they are
        octree_search.set_ray_tracers(
            RayTracers(
                root=[
                    ray_tracer if tracer is start_tracer else tracer
                    for tracer in search.ray_tracers
                ]
            )
        )

        result = InversionLocationResult()
        start = datetime_now()

        for waveform_image in self._images:
            events, trace = await octree_search.search(
                images=waveform_image,
                octree=search.octree.reset(),
                n_threads=0,
            )
            result.add_events(events)
            result.add_semblance_trace(trace)

        logger.info(
            "Located %d events in %s, cumulative semblance: %.2f",
            len(result.event),
            datetime_now() - start,
            result.cumulative_semblance,
        )

        return result

    async def invert(self, rundir: Path) -> InversionSummary:
        """Run the inversion and write its results to the rundir.

        Writes `iterations.jsonl` (one `InversionIteration` per trial),
        `best_model.json` (the best `LayeredModel`) and `results.json`
        (`InversionSummary`).

        Args:
            rundir (Path): Directory of the inversion run. Must exist.

        Returns:
            InversionSummary: Summary of the inversion.
        """
        started = datetime_now()
        await self.prepare(rundir=rundir)
        rng = np.random.default_rng(self.random_seed)

        iterations_file = rundir / "iterations.jsonl"
        iterations_file.unlink(missing_ok=True)

        def log_iteration(
            iteration: int,
            model: LayeredModelInversion,
            result: InversionLocationResult,
            accepted: bool,
        ) -> None:
            entry = InversionIteration(
                iteration=iteration,
                cumulative_semblance=result.cumulative_semblance,
                n_events=len(result.event),
                accepted=accepted,
                layers=model.layers,
            )
            with iterations_file.open("a") as file:
                file.write(entry.model_dump_json() + "\n")

        best_model = self.get_start_model()
        best_result = await self.test_velocity_model(best_model)
        log_iteration(0, best_model, best_result, accepted=True)
        start_semblance = best_result.cumulative_semblance
        best_iteration = 0
        n_accepted = 0

        for iteration in range(1, self.n_iterations + 1):
            model = LayeredModelInversion.from_layered_model(best_model)
            model.perturb_velocities(self.perturbation, rng=rng)
            result = await self.test_velocity_model(model)

            accepted = result.cumulative_semblance > best_result.cumulative_semblance
            log_iteration(iteration, model, result, accepted=accepted)
            if accepted:
                best_model, best_result = model, result
                best_iteration = iteration
                n_accepted += 1
            logger.info(
                "iteration %d/%d: semblance %.3f, best %.3f (iteration %d)",
                iteration,
                self.n_iterations,
                result.cumulative_semblance,
                best_result.cumulative_semblance,
                best_iteration,
            )

        (rundir / "best_model.json").write_text(best_model.model_dump_json(indent=2))
        summary = InversionSummary(
            n_iterations=self.n_iterations,
            n_accepted=n_accepted,
            n_selected_events=self.event_selection.n_selected_events(),
            start_semblance=start_semblance,
            best_semblance=best_result.cumulative_semblance,
            best_iteration=best_iteration,
            started=started,
            duration=datetime_now() - started,
        )
        (rundir / "results.json").write_text(summary.model_dump_json(indent=2))
        return summary
