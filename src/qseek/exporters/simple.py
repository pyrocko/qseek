from __future__ import annotations

import logging
from pathlib import Path

from rich import progress, prompt

from qseek.exporters.base import Exporter
from qseek.search import Search
from qseek.utils import time_to_path

logger = logging.getLogger(__name__)


class Simple(Exporter):
    """Export simple travel times in CSV format (E. Biondi, 2023)."""

    min_confidence_value: float = 0.3
    min_semblance_value: float = 0.5

    async def export(
        self,
        rundir: Path,
        outdir: Path,
    ) -> Path:
        logger.info("Export simple travel times in CSV format.")

        self.min_semblance_value = prompt.FloatPrompt.ask(
            "Minimum event semblance value",
            default=self.min_semblance_value,
        )
        self.min_confidence_value = prompt.FloatPrompt.ask(
            "Minimum pick confidence value",
            default=self.min_confidence_value,
        )

        search = Search.load_rundir(rundir)
        catalog = search.catalog

        traveltime_dir = outdir / "traveltimes"
        outdir.mkdir(parents=True)
        traveltime_dir.mkdir()

        search.stations.export_csv(outdir / "stations.csv")
        await catalog.export_csv(outdir / "events.csv")
        (outdir / "simple-export.json").write_text(self.model_dump_json(indent=2))

        n_observed_arrivals = 0
        n_events = 0

        for ev in progress.track(
            catalog,
            description="Exporting travel times",
            total=catalog.n_events,
        ):
            traveltime_file = traveltime_dir / f"{time_to_path(ev.time)}.csv"

            if ev.semblance < self.min_semblance_value:
                continue

            observed_arrivals = [
                (receiver, phase, arrival)
                for receiver in ev.receivers
                for phase, arrival in receiver.phase_arrivals.items()
                if arrival.observed is not None
                and arrival.observed.detection_value > self.min_confidence_value
            ]

            if not observed_arrivals:
                continue

            n_observed_arrivals += len(observed_arrivals)
            n_events += 1

            with traveltime_file.open("w") as file:
                file.write(f"# event_id: {ev.uid}\n")
                file.write(f"# event_time: {ev.time}\n")
                file.write(f"# event_lat: {ev.lat}\n")
                file.write(f"# event_lon: {ev.lon}\n")
                file.write(f"# event_depth: {ev.effective_depth}\n")
                file.write(f"# event_semblance: {ev.semblance}\n")
                file.write("# traveltime observations:\n")
                file.write(
                    "lat,lon,elevation,network,station,location,phase,confidence,traveltime\n"
                )

                for receiver, phase, arrival in observed_arrivals:
                    traveltime = arrival.observed.time - ev.time
                    file.write(
                        f"{receiver.lat},{receiver.lon},{receiver.effective_elevation},{receiver.network},{receiver.station},{receiver.location},{phase},{arrival.observed.detection_value},{traveltime.total_seconds()}\n",
                    )

        logger.info(
            "Exported %d observed arrivals from %d events.",
            n_observed_arrivals,
            n_events,
        )

        return outdir
