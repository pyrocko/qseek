from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rich
from rich.prompt import FloatPrompt, IntPrompt

from qseek.exporters.base import Exporter
from qseek.models.detection import EventDetection, PhaseDetection, Receiver
from qseek.models.station import Location, Station
from qseek.search import Search

logger = logging.getLogger(__name__)

CONTROL_FILE_TPL = """velest parameters are below, please modify according to their documents
{ref_lat}   {ref_lon}      0            0.0      0     0.00      1
{n_earthquakes}      0      0.0
{isingle:1d}     0
{max_distance_station}  0      {min_depth}    0.20    5.00    {use_station_correction:1d}
2      0.75      {vp_vs_ratio}       1
0.01    0.01      0.01    {velocity_damping}     {station_correction_damping}
1       0       0        {use_elevation:1d}        {use_station_correction:1d}
1         1         2        0
0         0         0         0         0         0        0
0.001   {iteration_number}   {invertratio}
{model_file}
stations_velest.sta

regionsnamen.dat
regionskoord.dat


{phase_file}

{mainout_file}
{outcheck_file}
{finalcnv_file}
{stacorrection_file}
"""


class VelestControlFile(NamedTuple):
    ref_lat: float
    ref_lon: float  # should be negative for East
    n_earthquakes: int
    isingle: bool = True
    min_depth: float = -0.2
    vp_vs_ratio: float = 1.65
    iteration_number: int = 99
    invertratio: int = 0
    model_file: str = "model.mod"
    phase_file: str = "phase_velest.pha"
    mainout_file: str = "main.out"
    outcheck_file: str = "log.out"
    finalcnv_file: str = "final.cnv"
    stacorrection_file: str = "stacor.dat"
    velocity_damping: float = 1.0  # Damping parameter for the velocity
    station_correction_damping: float = 0.1  # Damping parameter for the station
    max_distance_station: float = 200.0
    use_elevation: bool = False
    use_station_correction: bool = False
    allow_low_velocity: bool = False

    def write_config_file(self, file: Path):
        file.write_text(CONTROL_FILE_TPL.format(**self._asdict()))


class Velest(Exporter):
    """Crate a VELEST project folder for 1D velocity model estimation."""

    min_event_semblance: float = 0.2
    min_receivers_number: int = 10
    min_p_phase_confidence: float = 0.3
    min_s_phase_confidence: float = 0.3
    max_traveltime_delay: float = 2.5
    distance_border: float = 500.0
    n_picks_p: int = 0
    n_picks_s: int = 0
    n_events: int = 0

    async def export(self, rundir: Path, outdir: Path) -> Path:
        rich.print("Exporting qseek search to VELEST project folder")
        min_event_semblance = self.min_event_semblance
        min_receivers_number = IntPrompt.ask(
            "Minimum number of receivers (P phase)", default=10
        )
        min_distance_to_border = FloatPrompt.ask(
            "Minimum distance to border(m)", default=500.0
        )
        min_p_phase_confidence = FloatPrompt.ask(
            "Minimum pick probability for P phase", default=0.3
        )
        min_s_phase_confidence = FloatPrompt.ask(
            "Minimum pick probability for S phase", default=0.3
        )
        max_traveltime_delay = self.max_traveltime_delay
        self.min_receivers_number = min_receivers_number
        self.min_p_phase_confidence = min_p_phase_confidence
        self.min_s_phase_confidence = min_s_phase_confidence

        outdir.mkdir()
        search = Search.load_rundir(rundir)
        phases = search.image_functions.get_phases()
        for phase in phases:
            if "P" in phase:
                phase_p = phase
            if "S" in phase:
                phase_s = phase

        catalog = search.catalog

        # export station file
        stations = search.stations.stations
        station_file = outdir / "stations_velest.sta"
        self.export_station(stations=stations, filename=station_file)

        # export phase file
        phase_file = outdir / "phase_velest.pha"
        n_earthquakes = 0
        for event in catalog:
            if event.semblance < min_event_semblance:
                continue
            if event.receivers.n_observations(phase_p) < min_receivers_number:
                continue
            if event.distance_border < min_distance_to_border:
                continue

            observed_arrivals: list[tuple[Receiver, PhaseDetection]] = []

            for receiver in event.receivers:
                for _phase, detection in receiver.phase_arrivals.items():
                    if detection.observed is None:
                        continue
                    observed = detection.observed
                    if (
                        detection.phase == phase_p
                        and observed.detection_value <= min_p_phase_confidence
                    ):
                        continue
                    if (
                        detection.phase == phase_s
                        and observed.detection_value <= min_s_phase_confidence
                    ):
                        continue
                    if (
                        detection.traveltime_delay.total_seconds()
                        > max_traveltime_delay
                    ):
                        continue
                    observed_arrivals.append((receiver, detection))

            countp, counts = self.export_phases_slim(
                phase_file, event, observed_arrivals
            )
            self.n_picks_p += countp
            self.n_picks_s += counts
            n_earthquakes += 1
        self.n_events = n_earthquakes

        # export control file
        control_file = outdir / "velest.cmn"
        control_file_parameters = VelestControlFile(
            ref_lat=search.octree.location.lat,
            ref_lon=-search.octree.location.lon,
            n_earthquakes=n_earthquakes,
        )
        control_file_parameters.write_config_file(control_file)
        # export velocity model file
        dep = search.ray_tracers.root[0].earthmodel.layered_model.profile("z")
        vp = search.ray_tracers.root[0].earthmodel.layered_model.profile("vp")
        vs = search.ray_tracers.root[0].earthmodel.layered_model.profile("vs")
        dep_velest = []
        vp_velest = []
        vs_velest = []
        km = 1000.0
        for d, vpi, vsi in zip(dep, vp, vs, strict=False):
            if float(d) / km not in dep_velest:
                dep_velest.append(float(d) / km)
                vp_velest.append(float(vpi) / km)
                vs_velest.append(float(vsi) / km)
        velmod_file = outdir / "model.mod"
        make_velmod_file(velmod_file, vp_velest, vs_velest, dep_velest)

        export_info = outdir / "export_info.json"
        export_info.write_text(self.model_dump_json(indent=2))
        return outdir

    def export_phases_slim(
        self,
        outfile: Path,
        event: EventDetection,
        observed_arrivals: list[tuple[Receiver, PhaseDetection]],
    ):
        mag = event.magnitude.average if event.magnitude is not None else 0.0
        lat, lon = velest_location(event)
        write_out = (
            f"{event.time:%y%m%d %H%M %S.%f}"[:-4]
            + f" {lat} {lon} {event.depth/1000:7.2f}  {mag:5.2f}\n"
        )
        count_p = 0
        count_s = 0
        for rec, dectection in observed_arrivals:
            quality_weight = (
                np.digitize(
                    dectection.observed.detection_value, [1.0, 0.8, 0.6, 0.4, 0.0]
                )
                - 1
            )
            if dectection.phase.endswith("P"):
                phase = "P"
                count_p += 1
            else:
                phase = "S"
                count_s += 1
            traveltime = (dectection.observed.time - event.time).total_seconds()
            write_out += f"  {rec.station:6s}  {phase:1s}   {quality_weight:1d}  {traveltime:7.2f}\n"
        write_out += "\n"
        if count_p == 0 and count_s == 0:
            logger.warning("Removing event {event.time}: No phases observed")
        if count_p or count_s:
            with outfile.open("a") as file:
                file.write(write_out)

        return count_p, count_s

    def export_station(self, stations: list[Station], filename: Path) -> None:
        with filename.open("w") as fpout:
            fpout.write("(a6,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)\n")
            station_index = 1
            for station in stations:
                lat, lon = velest_location(station)
                fpout.write(
                    f"{station.station:6s}{lat} {lon} {int(station.elevation):4d} 1 {station_index:3d}  0.00   0.00\n"
                )
                station_index += 1
            fpout.write("\n")


def velest_location(location: Location) -> str:
    if location.effective_lat < 0:
        velest_lat = f"{location.effective_lat:7.4f}S"
    else:
        velest_lat = f"{location.effective_lat:7.4f}N"
    if location.effective_lon < 0:
        velest_lon = f"{location.effective_lon:8.4f}W"
    else:
        velest_lon = f"{location.effective_lon:8.4f}E"
    return velest_lat, velest_lon


def make_velmod_file(modname: Path, vp: list[float], vs: list[float], dep: list[float]):
    nlayer = len(dep)
    vdamp = 1.0
    with modname.open("w") as fp:
        fp.write("initial 1D-model for velest\n")
        # the second line - indicate the number of layers for Vp
        fp.write(f"{nlayer}      vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)\n")
        # vp model
        for vel, depth in zip(vp, dep, strict=False):
            fp.write(f"{vel:5.2f}     {depth:7.2f}  {vdamp:7.3f}\n")
        # vs model
        fp.write("%3d\n" % nlayer)
        for vel, depth in zip(vs, dep, strict=False):
            fp.write(f"{vel:5.2f}     {depth:7.2f}  {vdamp:7.3f}\n")
