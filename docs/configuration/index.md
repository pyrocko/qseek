---
icon: lucide/settings-2
---

# Configuration

You configure a search in one JSON file. It names your stations, the waveform data, the velocity models and the search volume, and selects the modules for every step of the search. Qseek validates the file when the search starts and reports every invalid or missing field.

Create a configuration with all defaults:

```sh title="Create a configuration file"
qseek config > my-search.json
```

## Minimal configuration

This configuration reads waveforms from an SDS archive, annotates phases with PhaseNet and calculates travel times for a constant velocity. Replace the paths, the location and the bounds with your own.

```json title="my-search.json"
{
  "project_dir": ".",
  "stations": {
    "station_xmls": ["meta/stations.xml"]
  },
  "data_provider": {
    "provider": "SDSArchive",
    "archive": "data/sds"
  },
  "octree": {
    "location": {
      "lat": 52.38,
      "lon": 13.06
    },
    "root_node_size": 2000.0,
    "n_levels": 3,
    "east_bounds": [-10000.0, 10000.0],
    "north_bounds": [-10000.0, 10000.0],
    "depth_bounds": [0.0, 20000.0]
  },
  "image_function": {
    "image": "SeisBench",
    "model": "PhaseNet",
    "pretrained": "original",
    "phase_map": {
      "P": "constant:P",
      "S": "constant:S"
    }
  },
  "ray_tracers": [
    {
      "tracer": "ConstantVelocityTracer",
      "phase": "constant:P",
      "velocity": 5000.0
    },
    {
      "tracer": "ConstantVelocityTracer",
      "phase": "constant:S",
      "velocity": 2900.0
    }
  ],
  "detection_threshold": "MAD",
  "window_length": "PT5M"
}
```

Durations such as the `window_length` are ISO 8601 durations: `"PT5M"` is 5 minutes. The [conventions](conventions.md) explain these formats.

## Modules

Every top-level field of the configuration configures one module of the search. [How Qseek works](../concepts/how-it-works.md) explains how they work together.

| Field | Module | Configures |
| --- | --- | --- |
| `stations` | [Stations](stations.md) | Station metadata and excluded stations |
| `data_provider` | [Waveforms](waveforms.md) | SDS archive, Pyrocko Squirrel or SeedLink streams |
| `pre_processing` | [Pre-processing](pre-processing.md) | Resampling, filters and denoising |
| `image_function` | [Image functions](image-functions.md) | Phase annotation and picking |
| `ray_tracers` | [Ray tracers](ray-tracers.md) | Travel times for every phase |
| `octree` | [Search volume](search-volume.md) | Location, size and resolution of the search volume |
| `station_weights` | [Station weighting](distance-weighting.md) | Weights of the stations for every node |
| `station_corrections` | [Station corrections](station-corrections.md) | Travel time delays per station |
| `magnitudes` | [Magnitudes](magnitudes.md) | Local and moment magnitudes |
| `features` | [Event features](features.md) | Ground motions |
| `callbacks` | [Callbacks](callbacks.md) | Alerts and custom actions for new detections |

The [conventions](conventions.md) explain the formats of paths, times, durations, station codes and locations.

## The search

The remaining fields of the search set the detection and the processing.

```python exec='on'
from qseek.utils import json_example
from qseek.search import Search

print(json_example(Search()))
```

<div class="qs-config" markdown>

::: qseek.search.Search
    options:
      heading_level: 3

</div>
