from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pyrocko.cake import GradientLayer

from qseek.exporters.base import Exporter
from qseek.search import Search
from qseek.tracers.cake import CakeTracer
from qseek.tracers.constant_velocity import ConstantVelocityTracer
from qseek.tracers.fast_marching import FastMarchingTracer

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.tracers.base import RayTracer
    from qseek.tracers.utils import LayeredEarthModel1D
    from qseek.utils import NSL

logger = logging.getLogger(__name__)

KM = 1000.0
# HypoDD limits: layers in the control file, characters per line of the control
# file, characters of a station label, events in hypoDD.inc of the distribution
MAX_LAYERS = 30
MAX_LINE_LENGTH = 220
MAX_STATION_LABEL = 7
MAX_EVENTS = 6500
UNUSED = -999
# Top of the first layer in km, above all sources, see discretize_earthmodel
TOP_FIRST_LAYER = -1.0

PH2DT_TPL = """\
* ph2dt.inp, written by Qseek
*--- I/O FILES:
* filename of station input:
station.dat
* filename of phase data input:
phase.dat
*--- DATA SELECTION PARAMETERS:
* MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS
{min_weight:g} {max_distance:.1f} {max_separation:.2f} {max_neighbors:d} \
{min_links:d} {min_observations:d} {max_observations:d}
"""

HYPODD_TPL = """\
hypoDD_2
* hypoDD.inp, written by Qseek
*--- INPUT FILE SELECTION
* filename of cross-corr diff. time input (blank if not available):

* filename of catalog travel time input (blank if not available):
dt.ct
* filename of initial hypocenter input:
event.sel
* filename of station input:
station.sel
*--- OUTPUT FILE SELECTION
* filename of initial hypocenter output (if blank: output to hypoDD.loc):
hypoDD.loc
* filename of relocated hypocenter output (if blank: output to hypoDD.reloc):
hypoDD.reloc
* filename of station residual output (if blank: no output written):
hypoDD.sta
* filename of data residual output (if blank: no output written):
hypoDD.res
* filename of takeoff angle output (if blank: no output written):
hypoDD.src
*--- DATA SELECTION:
* IDAT IPHA DIST
2 3 {max_distance:.1f}
*--- EVENT CLUSTERING:
* OBSCC OBSCT MINDS MAXDS MAXGAP
0 {min_links:d} {UNUSED} {UNUSED} {UNUSED}
*--- SOLUTION CONTROL:
* ISTART ISOLV IAQ NSET
{istart:d} {isolv:d} {iaq:d} {nset:d}
*--- DATA WEIGHTING AND REWEIGHTING:
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP
{weighting}
*--- FORWARD MODEL SPECIFICATIONS:
* IMOD
{imod:d}
{model}
*--- CLUSTER/EVENT SELECTION:
* CID
0
* ID (event IDs to relocate, 8 per line; none for all events)
"""

RUN_SCRIPT = """\
#!/bin/sh
# Relocate the detections with HypoDD. Set HYPODD_BIN to the directory of the
# ph2dt and hypoDD binaries if they are not in the PATH.
set -e
cd "$(dirname "$0")"
BIN="${HYPODD_BIN:+$HYPODD_BIN/}"
"${BIN}ph2dt" ph2dt.inp
"${BIN}hypoDD" hypoDD.inp
"""

README = """\
# HypoDD project folder

Qseek detections of `{rundir}`, prepared for double-difference relocation with
[HypoDD](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) v2.1 (Waldhauser, 2001).

| File | Content |
| --- | --- |
| `phase.dat` | {n_events} detections with {n_picks_p} P and {n_picks_s} S picks, \
input to ph2dt |
| `station.dat` | {n_stations} stations |
| `ph2dt.inp` | ph2dt control file |
| `hypoDD.inp` | hypoDD control file, catalog differential times only (`IDAT=2`) |
| `event_ids.csv` | HypoDD event ID, Qseek detection UID and origin time |
| `stations.csv` | HypoDD station label and Qseek station code (NSL) |
| `velocity_model.csv` | The layered velocity model written to `hypoDD.inp` |
| `export_info.json` | Settings of the export |
| `run.sh` | Runs ph2dt and hypoDD |

## Run HypoDD

```sh
HYPODD_BIN=~/src/HypoDD/bin ./run.sh
```

ph2dt writes the differential times `dt.ct` and the initial locations
`event.sel`; hypoDD writes the relocations to `hypoDD.reloc`. Map the event IDs
in `hypoDD.reloc` back to the Qseek detections with `event_ids.csv`.

Check the condition number (`CND`) of the LSQR iterations in `hypoDD.log`. It
should be about 40 to 80; tune `DAMP` in `hypoDD.inp` if it is not.

Depths are in km below sea level. HypoDD places the top of the velocity model
at each station's elevation.
"""


class Ph2DTSettings(BaseModel):
    """Settings of ph2dt, which forms the event pairs and their differential times."""

    min_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum pick weight (`MINWGHT`).",
    )
    max_distance: PositiveFloat | None = Field(
        default=None,
        description="Maximum distance between an event pair and a station in m "
        "(`MAXDIST`). `null` uses the largest event-station distance plus 10%.",
    )
    max_separation: PositiveFloat = Field(
        default=5000.0,
        description="Maximum separation of an event pair in m (`MAXSEP`).",
    )
    max_neighbors: PositiveInt = Field(
        default=20,
        description="Maximum number of neighbors per event (`MAXNGH`).",
    )
    min_links: PositiveInt = Field(
        default=8,
        description="Minimum number of differential times that link two events "
        "as neighbors (`MINLNK`).",
    )
    min_observations: PositiveInt = Field(
        default=8,
        description="Minimum number of differential times of a saved event pair "
        "(`MINOBS`).",
    )
    max_observations: PositiveInt | None = Field(
        default=None,
        description="Maximum number of differential times per event pair "
        "(`MAXOBS`). `null` uses twice the number of stations.",
    )


class IterationSet(BaseModel):
    """Weighting of the catalog differential times for a set of iterations."""

    n_iterations: PositiveInt = Field(
        default=5,
        description="Number of iterations with these weights (`NITER`).",
    )
    weight_p: float = Field(
        default=1.0,
        description="A priori weight of the P differential times (`WTCTP`). "
        "`-999` excludes them.",
    )
    weight_s: float = Field(
        default=0.5,
        description="A priori weight of the S differential times (`WTCTS`). "
        "`-999` excludes them.",
    )
    max_residual: float | None = Field(
        default=None,
        description="Residual cutoff (`WRCT`): below 1 a static cutoff in s, from 1 "
        "a multiple of the residual standard deviation. `null` keeps all data.",
    )
    max_separation: PositiveFloat | None = Field(
        default=None,
        description="Maximum separation of the linked events in m (`WDCT`). "
        "`null` does not limit the separation.",
    )
    damping: PositiveFloat = Field(
        default=80.0,
        description="Damping of the LSQR solver (`DAMP`). Tune it for a condition "
        "number (`CND` in `hypoDD.log`) of about 40 to 80.",
    )

    def as_line(self) -> str:
        def opt(value: float | None, scale: float = 1.0) -> str:
            return str(UNUSED) if value is None else f"{value / scale:g}"

        return (
            f"{self.n_iterations:d} {UNUSED} {UNUSED} {UNUSED} {UNUSED} "
            f"{self.weight_p:g} {self.weight_s:g} {opt(self.max_residual)} "
            f"{opt(self.max_separation, KM)} {self.damping:g}"
        )


def _default_iterations() -> list[IterationSet]:
    return [
        IterationSet(n_iterations=5),
        IterationSet(n_iterations=5, max_residual=6.0, max_separation=4000.0),
        IterationSet(n_iterations=5, max_residual=4.0, max_separation=2000.0),
    ]


class HypoDDSettings(BaseModel):
    """Settings of hypoDD, which relocates the events."""

    max_distance: PositiveFloat | None = Field(
        default=None,
        description="Maximum distance between the centroid of a cluster and a "
        "station in m (`DIST`). `null` uses the `max_distance` of ph2dt.",
    )
    min_links: PositiveInt = Field(
        default=8,
        description="Minimum number of catalog links of an event pair to keep the "
        "events in one cluster (`OBSCT`). Should not exceed `min_links` of ph2dt.",
    )
    initial_locations: Literal["catalog", "centroid"] = Field(
        default="catalog",
        description="Start from the Qseek locations or from the cluster centroid "
        "(`ISTART`).",
    )
    solver: Literal["LSQR", "SVD"] = Field(
        default="LSQR",
        description="Least squares solver (`ISOLV`). SVD gives meaningful errors "
        "but is limited to about 200 events.",
    )
    remove_airquakes: bool = Field(
        default=True,
        description="Remove events that locate above the surface (`IAQ`).",
    )
    iterations: list[IterationSet] = Field(
        default_factory=_default_iterations,
        min_length=1,
        max_length=10,
        description="Sets of iterations with their weighting (`NSET`, at most 10).",
    )


class HypoDDLayer(NamedTuple):
    top: float  # km below sea level
    vp: float  # km/s
    vp_vs: float


class HypoDD(Exporter):
    """Create a HypoDD project folder for double-difference relocation."""

    min_semblance: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum semblance of an exported detection.",
    )
    min_picks: PositiveInt = Field(
        default=6,
        description="Minimum number of selected P and S picks of an exported "
        "detection.",
    )
    min_distance_border: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum distance of an exported detection to the border of "
        "the search volume in m.",
    )
    min_pick_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence of an exported pick. The confidence is the "
        "pick weight in `phase.dat`.",
    )
    max_residual: PositiveFloat = Field(
        default=1.0,
        description="Maximum absolute travel time residual of an exported pick to "
        "the modeled arrival in s.",
    )
    max_layer_thickness: PositiveFloat = Field(
        default=500.0,
        description="Gradient layers of the velocity model are split into constant "
        "velocity layers of this maximum thickness in m, down to the bottom of the "
        "search volume.",
    )
    ph2dt: Ph2DTSettings = Field(
        default_factory=Ph2DTSettings,
        description="Settings of ph2dt.",
    )
    hypodd: HypoDDSettings = Field(
        default_factory=HypoDDSettings,
        description="Settings of hypoDD.",
    )

    async def export(self, rundir: Path, outdir: Path) -> Path:
        logger.info("exporting detections of %s to HypoDD project folder", rundir)
        search = Search.load_rundir(rundir)

        events: list[tuple[int, EventDetection, datetime]] = []
        event_picks: list[list[tuple[NSL, float, float, str]]] = []
        stations: dict[NSL, tuple[float, float, float]] = {}

        for event in search.catalog:
            if event.semblance < self.min_semblance:
                continue
            if event.distance_border < self.min_distance_border:
                continue

            # HypoDD keeps the origin time with 10 ms resolution, the travel times
            # refer to the rounded origin time
            origin = round_time(event.time)
            picks: list[tuple[NSL, float, float, str]] = []
            for receiver in event.receivers:
                for phase, arrival in receiver.phase_arrivals.items():
                    observed = arrival.observed
                    phase_type = phase_hint(phase)
                    if observed is None or phase_type is None:
                        continue
                    if observed.detection_value < self.min_pick_confidence:
                        continue
                    delay = arrival.traveltime_delay
                    if delay is None or abs(delay.total_seconds()) > self.max_residual:
                        continue
                    picks.append(
                        (
                            receiver.nsl,
                            (observed.time - origin).total_seconds(),
                            observed.detection_value,
                            phase_type,
                        )
                    )
            if len(picks) < self.min_picks:
                continue

            for receiver in event.receivers:
                stations.setdefault(
                    receiver.nsl,
                    (
                        receiver.effective_lat,
                        receiver.effective_lon,
                        receiver.effective_elevation,
                    ),
                )
            events.append((len(events) + 1, event, origin))
            event_picks.append(picks)

        if not events:
            raise ValueError("no detections selected for the HypoDD export")
        if len(events) > MAX_EVENTS:
            logger.warning(
                "%d events exceed MAXEVE=%d of the HypoDD distribution,"
                " increase MAXEVE in include/hypoDD.inc",
                len(events),
                MAX_EVENTS,
            )

        labels = station_labels(list(stations))
        max_distance = self.ph2dt.max_distance or 1.1 * max_event_station_distance(
            [event for _, event, _ in events], list(stations.values())
        )

        layers, imod = self.get_velocity_model(search)

        outdir.mkdir(parents=True)
        n_picks = self.write_phases(outdir / "phase.dat", events, event_picks, labels)
        with (outdir / "station.dat").open("w") as file:
            for nsl, (lat, lon, elevation) in stations.items():
                file.write(f"{labels[nsl]} {lat:.6f} {lon:.6f} {elevation:.1f}\n")
        with (outdir / "stations.csv").open("w") as file:
            file.write("label,nsl,lat,lon,elevation\n")
            for nsl, (lat, lon, elevation) in stations.items():
                file.write(
                    f"{labels[nsl]},{nsl.pretty},{lat:.6f},{lon:.6f},{elevation:.1f}\n"
                )
        with (outdir / "event_ids.csv").open("w") as file:
            file.write("id,uid,time,hypodd_time\n")
            for event_id, event, origin in events:
                file.write(
                    f"{event_id},{event.uid},{event.time.isoformat()},"
                    f"{origin.isoformat()}\n"
                )
        with (outdir / "velocity_model.csv").open("w") as file:
            file.write("top_km,vp_km_s,vs_km_s,vp_vs\n")
            for layer in layers:
                file.write(
                    f"{layer.top:.3f},{layer.vp:.3f},"
                    f"{layer.vp / layer.vp_vs:.3f},{layer.vp_vs:.3f}\n"
                )

        max_observations = self.ph2dt.max_observations or 2 * len(stations)
        (outdir / "ph2dt.inp").write_text(
            PH2DT_TPL.format(
                min_weight=self.ph2dt.min_weight,
                max_distance=max_distance / KM,
                max_separation=self.ph2dt.max_separation / KM,
                max_neighbors=self.ph2dt.max_neighbors,
                min_links=self.ph2dt.min_links,
                min_observations=self.ph2dt.min_observations,
                max_observations=max_observations,
            )
        )
        hypodd = self.hypodd
        (outdir / "hypoDD.inp").write_text(
            HYPODD_TPL.format(
                max_distance=(hypodd.max_distance or max_distance) / KM,
                min_links=hypodd.min_links,
                UNUSED=UNUSED,
                istart=1 if hypodd.initial_locations == "centroid" else 2,
                isolv=1 if hypodd.solver == "SVD" else 2,
                iaq=int(hypodd.remove_airquakes),
                nset=len(hypodd.iterations),
                weighting="\n".join(it.as_line() for it in hypodd.iterations),
                imod=imod,
                model=model_block(layers),
            )
        )
        run_script = outdir / "run.sh"
        run_script.write_text(RUN_SCRIPT)
        run_script.chmod(0o755)

        (outdir / "README.md").write_text(
            README.format(
                rundir=rundir.name,
                n_events=len(events),
                n_picks_p=n_picks["P"],
                n_picks_s=n_picks["S"],
                n_stations=len(stations),
            )
        )
        (outdir / "export_info.json").write_text(self.model_dump_json(indent=2))

        logger.info(
            "exported %d events with %d P and %d S picks at %d stations to %s",
            len(events),
            n_picks["P"],
            n_picks["S"],
            len(stations),
            outdir,
        )
        return outdir

    def write_phases(
        self,
        file: Path,
        events: list[tuple[int, EventDetection, datetime]],
        event_picks: list[list[tuple[NSL, float, float, str]]],
        labels: dict[NSL, str],
    ) -> dict[str, int]:
        """Write the phase file for ph2dt.

        Returns:
            dict[str, int]: Number of written P and S picks.
        """
        n_picks = {"P": 0, "S": 0}
        n_shallow = 0
        lines = []
        for (event_id, event, origin), picks in zip(events, event_picks, strict=True):
            depth = event.effective_depth / KM
            if depth < 0.0:
                n_shallow += 1
                depth = 0.0
            uncertainty = event.uncertainty
            magnitude = event.magnitude
            mag = magnitude.average if magnitude and magnitude.average else 0.0
            seconds = origin.second + origin.microsecond / 1e6
            lines.append(
                f"# {origin:%Y %m %d %H %M} {seconds:.2f}"
                f" {event.effective_lat:.6f} {event.effective_lon:.6f} {depth:.4f}"
                f" {mag:.2f}"
                f" {uncertainty.horizontal / KM if uncertainty else 0.0:.4f}"
                f" {uncertainty.vertical / KM if uncertainty else 0.0:.4f}"
                f" {event.rms or 0.0:.4f} {event_id:d}"
            )
            for nsl, traveltime, weight, phase_type in picks:
                lines.append(
                    f"{labels[nsl]} {traveltime:.3f} {weight:.3f} {phase_type}"
                )
                n_picks[phase_type] += 1
        file.write_text("\n".join(lines) + "\n")

        if n_shallow:
            logger.warning(
                "%d events above sea level, set to 0 km depth for HypoDD",
                n_shallow,
            )
        return n_picks

    def get_velocity_model(self, search: Search) -> tuple[list[HypoDDLayer], int]:
        """Get the HypoDD velocity model from the ray tracer of the P phase.

        Returns:
            tuple[list[HypoDDLayer], int]: The layers and the HypoDD model type
                `IMOD`: 1 for layers with variable Vp/Vs ratio, 5 for a constant
                velocity.
        """
        phases = search.ray_tracers.get_available_phases()
        phase_p = next((ph for ph in phases if phase_hint(ph) == "P"), None)
        if phase_p is None:
            raise ValueError("no ray tracer for a P phase found")
        tracer = search.ray_tracers.get_phase_tracer(phase_p)

        if isinstance(tracer, ConstantVelocityTracer):
            phase_s = next((ph for ph in phases if phase_hint(ph) == "S"), None)
            tracer_s = search.ray_tracers.get_phase_tracer(phase_s) if phase_s else None
            if isinstance(tracer_s, ConstantVelocityTracer):
                vp_vs = tracer.velocity / tracer_s.velocity
            else:
                vp_vs = math.sqrt(3.0)
                logger.warning("no constant S velocity found, using Vp/Vs = %g", vp_vs)
            return [HypoDDLayer(0.0, tracer.velocity / KM, vp_vs)], 5

        earthmodel = get_earthmodel(tracer)
        max_depth = search.octree.effective_depth_bounds.end
        layers = discretize_earthmodel(
            earthmodel,
            max_thickness=self.max_layer_thickness,
            max_depth=max_depth,
        )
        if len(layers) > MAX_LAYERS:
            raise ValueError(
                f"the velocity model has {len(layers)} layers, HypoDD allows"
                f" {MAX_LAYERS}; increase max_layer_thickness"
            )
        return layers, 1


def round_time(time: datetime, resolution: float = 0.01) -> datetime:
    """Round a time to the given resolution in seconds."""
    micro = int(resolution * 1e6)
    rounded = round(time.microsecond / micro) * micro
    return time.replace(microsecond=0) + timedelta(microseconds=rounded)


def phase_hint(phase: str) -> Literal["P", "S"] | None:
    """Get the HypoDD phase type, `P` or `S`, of a phase description.

    Args:
        phase: Phase description, e.g. `cake:P` or `fm:S`.

    Returns:
        `P` or `S`, or `None` if the phase is neither.
    """
    name = phase.rsplit(":", 1)[-1][:1].upper()
    if name == "P":
        return "P"
    if name == "S":
        return "S"
    return None


def station_labels(nsls: list[NSL]) -> dict[NSL, str]:
    """Get unique HypoDD station labels of up to 7 characters.

    The label is the station code, or network and station code if the station code
    is not unique. Stations that are still not unique get a running number.
    """
    labels: dict[NSL, str] = {}
    station_counts: dict[str, int] = {}
    for nsl in nsls:
        station_counts[nsl.station] = station_counts.get(nsl.station, 0) + 1

    used: set[str] = set()
    for nsl in nsls:
        label = nsl.station
        if station_counts[nsl.station] > 1:
            label = f"{nsl.network}{nsl.station}"
        label = label[:MAX_STATION_LABEL]
        if label in used:
            for idx in range(100):
                candidate = f"{label[: MAX_STATION_LABEL - 2]}{idx:02d}"
                if candidate not in used:
                    label = candidate
                    break
        used.add(label)
        labels[nsl] = label
    return labels


def max_event_station_distance(
    events: list[EventDetection],
    stations: list[tuple[float, float, float]],
) -> float:
    """Largest epicentral distance between an event and a station in m."""
    from pyrocko import orthodrome as od

    return max(
        od.distance_accurate50m(event.effective_lat, event.effective_lon, lat, lon)
        for event in events
        for lat, lon, _ in stations
    )


def get_earthmodel(tracer: RayTracer) -> LayeredEarthModel1D:
    """Get the 1D velocity model of a ray tracer."""
    if isinstance(tracer, CakeTracer):
        earthmodel = tracer.earthmodel
    elif isinstance(tracer, FastMarchingTracer) and tracer.velocity_model:
        earthmodel = tracer.velocity_model
    else:
        raise TypeError(
            f"cannot export the velocity model of {tracer.__class__.__name__},"
            " HypoDD needs a 1D layered model"
        )
    filename = earthmodel.filename
    if earthmodel.raw_file_data is None and filename and not filename.exists():
        raise FileNotFoundError(
            f"velocity model {filename} not found, run the export from the"
            " directory of the search configuration"
        )
    return earthmodel


def discretize_earthmodel(
    earthmodel: LayeredEarthModel1D,
    max_thickness: float,
    max_depth: float,
) -> list[HypoDDLayer]:
    """Convert a 1D velocity model to constant velocity layers for HypoDD.

    Gradient layers above `max_depth` are split into layers of at most
    `max_thickness`, deeper gradient layers into one layer. Each layer gets the
    harmonic mean velocity of its depth range, which keeps the vertical travel time.
    The part of the model above sea level is merged into the top layer. HypoDD
    places the top of the model at the station elevation, the top of the first layer
    is written as `TOP_FIRST_LAYER` above sea level: hypoDD v2.1 looks up the
    velocity at the source in the layer above the first layer top that is not
    shallower than the source, a source at the top of the first layer gets an
    undefined velocity and the inversion fails with NaN.

    Args:
        earthmodel: The velocity model.
        max_thickness: Maximum thickness of the split gradient layers in m.
        max_depth: Depth below sea level in m down to which gradient layers are
            split.

    Returns:
        list[HypoDDLayer]: Layers with their top depth in km, Vp in km/s and Vp/Vs.
    """
    intervals: list[tuple[float, float, float]] = []
    for layer in earthmodel.layered_model.layers():
        ztop, zbot = max(layer.ztop, 0.0), layer.zbot
        if zbot <= 0.0:
            continue
        if isinstance(layer, GradientLayer):
            thickness = layer.zbot - layer.ztop
            vp_top, vs_top = layer.mtop.vp, layer.mtop.vs
            grad_vp = (layer.mbot.vp - vp_top) / thickness
            grad_vs = (layer.mbot.vs - vs_top) / thickness

            bounds = [ztop]
            split_bottom = min(zbot, max_depth)
            if split_bottom > ztop:
                n_split = math.ceil((split_bottom - ztop) / max_thickness)
                bounds += [
                    ztop + (split_bottom - ztop) * (i + 1) / n_split
                    for i in range(n_split)
                ]
            if zbot > bounds[-1]:
                bounds.append(zbot)

            for z0, z1 in itertools.pairwise(bounds):
                vp = harmonic_mean(vp_top, grad_vp, z0 - layer.ztop, z1 - layer.ztop)
                vs = harmonic_mean(vs_top, grad_vs, z0 - layer.ztop, z1 - layer.ztop)
                intervals.append((z0, vp, vs))
        else:
            intervals.append((ztop, layer.m.vp, layer.m.vs))

    layers: list[HypoDDLayer] = []
    for top, vp, vs in intervals:
        if vs <= 0.0:
            raise ValueError(f"the velocity model has no S velocity at {top} m")
        if layers and math.isclose(top / KM, layers[-1].top):
            layers.pop()  # zero thickness, e.g. the clipped part above sea level
        layer = HypoDDLayer(round(top / KM, 3), round(vp / KM, 3), round(vp / vs, 3))
        if layers and layers[-1][1:] == layer[1:]:
            continue
        layers.append(layer)
    return [layers[0]._replace(top=TOP_FIRST_LAYER), *layers[1:]]


def harmonic_mean(v0: float, gradient: float, z0: float, z1: float) -> float:
    """Harmonic mean of a linear velocity profile `v0 + gradient * z` from z0 to z1."""
    va, vb = v0 + gradient * z0, v0 + gradient * z1
    if math.isclose(va, vb):
        return (va + vb) / 2
    return (vb - va) / math.log(vb / va)


def model_block(layers: list[HypoDDLayer]) -> str:
    """Model block of hypoDD.inp for IMOD 1 and 5."""
    lines = {
        "TOP": " ".join(f"{layer.top:.3f}" for layer in layers),
        "VELP": " ".join(f"{layer.vp:.3f}" for layer in layers),
        "RAT": " ".join(f"{layer.vp_vs:.3f}" for layer in layers),
    }
    block = []
    for name, line in lines.items():
        if len(line) >= MAX_LINE_LENGTH:
            raise ValueError(
                f"line {name} of the velocity model exceeds {MAX_LINE_LENGTH}"
                " characters, increase max_layer_thickness"
            )
        block += [f"* {name}:", line]
    return "\n".join(block)
