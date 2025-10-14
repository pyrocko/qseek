from __future__ import annotations

import asyncio
import logging
from cProfile import Profile
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from qseek.tracers.fast_marching import FastMarchingTracer
from qseek.tracers.utils import LayeredEarthModel1D
from qseek.utils import setup_rich_logging
from qseek_inversion.inversion import InversionLayered1D
from qseek_inversion.waveforms.event_selection import EventWaveformsSelection
from qseek_inversion.waveforms.synthetics import SyntheticEvents

p = Profile()
setup_rich_logging(logging.INFO)

import_rundir = Path(
    "/home/marius/Projects/2025-copahue/qseek/copahue-fmm.bak-2025-10-06T192002"
)
inv2 = InversionLayered1D(
    import_rundir=import_rundir,
    event_selection=EventWaveformsSelection(
        import_rundir=import_rundir,
        number_events=50,
        events_per_batch=25,
        seconds_before_event=10.0,
        seconds_after_event=5.0,
        taper=0.1,
    ),
)

model = LayeredEarthModel1D(
    filename=import_rundir.parent / "stratovolcano.nd",
)

inv = InversionLayered1D(
    import_rundir=import_rundir,
    event_selection=SyntheticEvents(
        ray_tracer=FastMarchingTracer(
            velocity_model=model,
            implementation="pyrocko",
        ),
        import_rundir=import_rundir,
        n_events=10,
        inter_event_time=timedelta(seconds=30),
    ),
)


async def run(export: bool = True):
    # inv = inv2
    await inv.prepare()
    for img in inv._images:
        break
        img.snuffle()
    model = inv.get_start_model()

    results = []
    # yappi.start()
    for _ in range(1):
        res = await inv.test_velocity_model(model)
        results.append(res)
        # model.perturb_velocities(noise=0.1)

    # yappi.get_func_stats().save("/tmp/inversion.prof", type="pstat")

    for res in results:
        logging.info("Cumulative semblance: %.2f", res.cumulative_semblance)

    # snuffle([res._trace for res in results])

    if export:
        inv_path = Path("/tmp/events_inversion.csv")
        with NamedTemporaryFile("w") as tmpfile:
            tmpfile = Path(tmpfile.name).with_suffix(".csv")
            for ev in res.event:
                ev.export_csv_line(tmpfile)
            tmpfile.rename(inv_path)


asyncio.run(run())
