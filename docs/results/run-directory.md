---
icon: lucide/folder-open
---

# Run directory

Every search writes its results into a run directory, named after the configuration file: `qseek search my-search.json` creates `my-search/`. Qseek adds each detection while the search runs, so you can look at the results before the search ends.

```text
my-search/
├── search.json                     # the configuration of the run
├── progress.json                   # how far the search got, for `qseek continue`
├── results.json                    # statistics of the finished search
├── qseek.log                       # the log of the search
├── detections.jsonl                # all detections, one JSON object per line
├── detections_receivers.jsonl      # modeled and picked arrivals of every detection
├── semblance.mseed                 # the detection function over time
├── csv/
│   ├── detections.csv              # the detections as a table
│   ├── detections_jittered.csv     # the same, with jittered locations
│   └── stations.csv                # the stations of the search
├── pyrocko_detections.list         # the detections as Pyrocko events
├── pyrocko_detections_jittered.list
├── pyrocko_stations.yaml           # the stations as Pyrocko stations
└── pyrocko_markers/                # event and phase markers for Snuffler
```

With `save_images`, Qseek also writes the phase images into `images/`. Searches with 3D velocity models export the models to `3d-models/`, see [visualize 3D models](../configuration/ray-tracers.md#visualize-3d-models). `pyrocko_markers/` holds one file per detection with its event marker and the modeled and picked phase markers, which `qseek snuffler` shows with the waveforms.

## Scripts and agents

`qseek --non-interactive search` prints only errors and `key: value` lines: `qseek`, `rundir`, `log`, `progress`, `config`, `results`, `catalog`, `duration`, `detections` and `status`. A failed run prints `status: failed` and `error:`, and exits with 2 for a configuration or rundir problem, 1 for any other error and 130 when interrupted. `progress.json` holds the processed percentage, the number of events and the remaining time, and is updated after every batch.

`qseek search --check config.json` validates the configuration and the available stations and waveform data without creating a run directory. `qseek --non-interactive summary my-search` prints the paths and the state of an existing run.

## Detections

`detections.jsonl` holds all detections in the [JSON Lines](https://jsonlines.org/) format, one detection per line. The modeled and picked arrivals at every station are in `detections_receivers.jsonl`, in the same order.

`csv/detections.csv` is a table of the detections, for spreadsheets, GIS software and plotting:

| Column | Description |
| --- | --- |
| `time` | Origin time, ISO 8601 in UTC |
| `lat`, `lon` | Epicenter in degrees |
| `depth` | Depth in meters, positive down |
| `east_shift`, `north_shift` | Location relative to the center of the search volume in meters |
| `distance_border` | Distance to the border of the search volume in meters |
| `semblance` | Peak value of the detection function |
| `azimuthal_coverage` | 360° minus the largest azimuthal gap between stations with picks, in degrees |
| `n_stations`, `n_picks` | Number of stations and of phase picks |
| `rms` | Root mean square of the travel time residuals of the picks in seconds, averaged over the phases |
| `uncertainty_horizontal`, `uncertainty_vertical` | Location uncertainty in meters |
| `WKT_geom` | The location as a WKT point, for GIS software |

Every magnitude adds its own columns.

!!! note "Jittered locations"
    The locations of the detections fall onto the nodes of the octree, which can show up as a grid pattern in maps. The `_jittered` files shift every location randomly by up to half the smallest node size in each direction, for maps and density plots. Use the files without jitter for analysis.

## Read the detections in Python

```python title="Load the detections of a run"
from pathlib import Path

from qseek.models.catalog import EventCatalog

catalog = EventCatalog.load_rundir(Path("my-search"))
print(f"{catalog.n_events} detections")

for detection in catalog:
    lat, lon = detection.effective_lat_lon
    print(detection.time, lat, lon, detection.effective_depth, detection.semblance)
```

Each detection also holds its `magnitudes`, `features` and location `uncertainty`, see the [detections reference](../reference/api/detections.md).

## Next steps

[Explore the results](explore.md) in the web UI, in Snuffler or in GIS software, or export them to other formats.
