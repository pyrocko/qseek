# Getting Started

## Installation

The installation is straight-forward:

```sh
pip install lassie-v2
```

or install from GitHub

```sh title="From GitHub"
pip install git+https://github.com/pyrocko/lassie-v2
```

## Initializing a new Project

Once installed you can run the lassie executeable

```sh title="Initialize new Project"
lassie init my-project
```

Check out the `search.json` config file and add your waveform data and velocity models.

??? abstract "Minimal Configuration Example"
    Here is a minimal JSON configuration for Lassie
    ```json
    {
    "project_dir": ".",
    "stations": {
        "station_xmls": [],
        "pyrocko_station_yamls": ["search/pyrocko-stations.yaml"],
    },
    "data_provider": {
        "provider": "PyrockoSquirrel",
        "environment": ".",
        "waveform_dirs": ["data/"],
    },
    "octree": {
        "location": {
        "lat": 0.0,
        "lon": 0.0,
        "east_shift": 0.0,
        "north_shift": 0.0,
        "elevation": 0.0,
        "depth": 0.0
        },
        "size_initial": 2000.0,
        "size_limit": 500.0,
        "east_bounds": [
        -10000.0,
        10000.0
        ],
        "north_bounds": [
        -10000.0,
        10000.0
        ],
        "depth_bounds": [
        0.0,
        20000.0
        ],
        "absorbing_boundary": 1000.0
    },
    "image_functions": [
        {
        "image": "PhaseNet",
        "model": "ethz",
        "torch_use_cuda": false,
        "phase_map": {
            "P": "constant:P",
            "S": "constant:S"
        },
        "weights": {
            "P": 1.0,
            "S": 1.0
        }
        }
    ],
    "ray_tracers": [
        {
        "tracer": "ConstantVelocityTracer",
        "phase": "constant:P",
        "velocity": 5000.0
        }
    ],
    "station_corrections": {
        "rundir": null,
        "measure": "median",
        "weighting": "mul-PhaseNet-semblance",
        "minimum_num_picks": 5,
        "minimum_distance_border": 2000.0,
        "minimum_depth": 3000.0
    },
    "event_features": [],
    "sampling_rate": 100,
    "detection_threshold": 0.05,
    "detection_blinding": "PT2S",
    "image_mean_p": 1.0,
    "node_split_threshold": 0.9,
    "window_length": "PT300S",
    "n_threads_parstack": 0,
    "n_threads_argmax": 4,
    "plot_octree_surface": false,
    "created": "2023-10-29T19:17:17.676279Z"
    }
    ```


## Starting the Search
Once happy, start the search with

```sh title="Start earthquake detection"
lassie search search.json
```
