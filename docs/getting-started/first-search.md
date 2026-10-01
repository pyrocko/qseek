---
icon: lucide/play
---

# Set up your own search

You run Qseek with the `qseek` command and a JSON configuration file. The configuration describes your stations, waveform data, velocity model and search volume. The [quick start](quick-start.md) shows a complete example.

## Create a configuration

Print the default configuration into a new file:

```sh title="Create a configuration file"
qseek config > my-search.json
```

Open `my-search.json` and set at least these fields; everything else keeps its default:

- `stations`: your [station metadata](../configuration/stations.md).
- `data_provider`: your [waveform data](../configuration/waveforms.md).
- `octree`: the [search volume](../configuration/search-volume.md) around your seismicity.
- `image_function` with its `phase_map`: the [phase annotation](../configuration/image-functions.md).
- `ray_tracers`: the [travel times](../configuration/ray-tracers.md) in your velocity model.

The [minimal configuration](../configuration/index.md#minimal-configuration) shows these fields together, and the [configuration overview](../configuration/index.md) explains every module.

??? quote "Default configuration"
    ```bash exec='on' result='json'
    qseek config | sed -e "s|$PWD|.|g" -e "s|$HOME|~|g"
    ```

## Start the search

Start the search in the directory of the configuration file, since Qseek resolves relative paths from the directory where you start it:

```sh title="Start the search"
qseek search my-search.json
```

Qseek creates the run directory `my-search/`, named after the configuration file, in the `project_dir` of the configuration. `project_dir` defaults to `.`, the directory where you start the search. Qseek writes the detections into the run directory while the search runs; if the search stops, [continue it](../guides/manage-runs.md#continue-a-search).

When the search has detections, look at the [run directory](../results/run-directory.md) and [explore the results](../results/explore.md). The [command line reference](../reference/cli.md) lists all `qseek` commands.
