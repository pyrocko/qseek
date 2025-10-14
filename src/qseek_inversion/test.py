import asyncio
from pathlib import Path

from pyrocko.squirrel import Squirrel
from pyrocko.trace import snuffle

from qseek.models.catalog import EventCatalog
from qseek_inversion.event_waveforms import EventWaveformsSelection

sq = Squirrel(env="/home/marius/Projects/2025-copahue/data")
sq.add("/home/marius/Projects/2025-copahue/data/waveforms")

rundir = Path("/home/marius/Projects/2025-copahue/qseek/copahue-fmm/")
cat = EventCatalog.load_rundir(rundir)

ew = EventWaveformsSelection()

batch = asyncio.run(ew.get_batch(cat, sq))

snuffle(batch.traces)
