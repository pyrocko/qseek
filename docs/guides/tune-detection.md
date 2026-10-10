---
icon: lucide/sliders-horizontal
---

# Tune the detection

A first search with the defaults tells you a lot about your data. This guide shows how to check the results and which settings to change for too few detections, too many false detections or imprecise locations.

## Check the results

- **Waveforms and picks:** open the run in Snuffler and compare the modeled and picked arrivals with the waveforms of a few detections.

    ```sh
    qseek snuffler my-search/ --show-observed --show-modelled --show-semblance
    ```

- **Detection function:** `--show-semblance` adds the detection function, the maximum semblance over all nodes. Clear peaks above a flat noise level mean the stack works.
- **Quality columns:** `csv/detections.csv` has the `semblance`, the number of picks `n_picks`, the travel time residuals `rms` and the location uncertainty of every detection. Plot them to find a threshold that separates events from noise. See the [run directory](../results/run-directory.md#detections).

## Too few detections

- **Threshold:** the default [trigger](../configuration/triggers.md) detects peaks of the detection function above a semblance of 0.3. Lower the `threshold` below the semblance of the weakest events you see, or use an adaptive trigger, which lowers the threshold in quiet windows.
- **Image function:** try pre-trained weights that fit your data, e.g. trained on a similar region or instrument type, or another model. See [image functions](../configuration/image-functions.md).
- **Frequency band:** adjust the bandpass of the [pre-processing](../configuration/pre-processing.md) to the frequencies of your events, e.g. higher frequencies for microseismicity.
- **Minimum stations:** lower [`min_stations`][qseek.search.Search.min_stations] for small networks.

## Too many false detections

- **Threshold:** raise the `threshold` of the trigger above the semblance of the noise.
- **Picks:** filter the detections by `n_picks`. Noise rarely produces consistent picks at many stations.
- **Distance weighting:** with the default [distance weights](../configuration/distance-weighting.md), each node relies on its closest stations. Noise at a single close station can dominate small networks; increase [`required_closest_stations`][qseek.distance_weights.DistanceWeights.required_closest_stations].

## Imprecise locations

- **Velocity model:** the travel times need a velocity model that fits your region. A 1D model of the region is better than a constant velocity, a 3D model better still in complex geology. See [ray tracers](../configuration/ray-tracers.md).
- **Station corrections:** extract station corrections from a first run and search again. They correct what the velocity model misses, see [station corrections](../configuration/station-corrections.md).
- **Resolution:** increase [`n_levels`][qseek.octree.Octree.n_levels] to the [search volume](../configuration/search-volume.md) for smaller nodes, and keep [`node_interpolation`][qseek.search.Search.node_interpolation] on.
- **Boundary:** detections at the border of the volume are ignored. If the seismicity reaches the border, enlarge the volume.

## Repeated detections of one event

The `blinding` of the [trigger](../configuration/triggers.md) sets the minimum time between two detections, 1 s by default. Raise it if a long or complex event is detected twice.
