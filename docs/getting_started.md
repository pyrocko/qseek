# Getting Started

## Installation

The installation is straight-forward:

```sh title="From GitHub"
pip install git+https://github.com/pyrocko/lassie-v2
```

## Running Lassie

The main entry point in the executeable is the `lassie` command. The provided command line interface (CLI) and a JSON config file is all what is needed to run the program.

```bash exec='on' result='ansi' source='above'
lassie -h
```

## Initializing a New Project

Once installed you can run the lassie executeable to initialize a new project.

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
      }
    ],
    "ray_tracers": [
      {
      "tracer": "ConstantVelocityTracer",
      "phase": "constant:P",
      "velocity": 5000.0
      }
    ],
    "station_corrections": {},
    "event_features": [],
    "sampling_rate": 100,
    "detection_threshold": 0.05,
    "detection_blinding": "PT2S",
    "node_split_threshold": 0.9,
    "window_length": "PT300S",
    "n_threads_parstack": 0,
    "n_threads_argmax": 4,
  }
  ```

For more details and information about the component, head over to [details of the modules](components/seismic_data.md).

## Starting the Search

Once happy, start the lassie CLI.

```sh title="Start earthquake detection"
lassie search search.json
```
