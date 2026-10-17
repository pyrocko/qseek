---
icon: lucide/zap
---

# Performance

Qseek is built for large networks and years of data. The [benchmark](../about/benchmark.md) shows the throughput for different network sizes.

These settings decide how fast your search runs.

## Phase annotation on the GPU

The phase annotation with machine learning pickers runs much faster on a GPU. Run it on a CUDA GPU with [`torch_use_cuda`][qseek.images.seisbench.SeisBench.torch_use_cuda] of the [SeisBench image function](../configuration/image-functions.md#seisbench): `true` uses the default device, a number selects a device, e.g. `0` for the first one.

```json title="Annotation on the GPU"
"image_function": {
  "image": "SeisBench",
  "model": "PhaseNet",
  "torch_use_cuda": true,
  "batch_size": 128
}
```

A larger [`batch_size`][qseek.images.seisbench.SeisBench.batch_size] can improve the throughput on the GPU. Without a GPU, [`torch_cpu_threads`][qseek.images.seisbench.SeisBench.torch_cpu_threads] sets the number of CPU threads for the annotation.

## Stacking and migration

- **Threads:** [`n_threads`][qseek.search.Search.n_threads] of the search sets the threads for stacking and migration. The default `"auto"` uses the available cores and leaves resources for loading the data and the annotation.
- **Search volume:** Qseek stacks every root node in every window. Fewer, larger root nodes make the search faster; add octree levels instead of shrinking the root nodes. See [search volume](../configuration/search-volume.md).
- **Station weights:** stations with a weight of zero are skipped in the stack. The [station weights](../configuration/station-weights.md) limit large networks to the stations close to each node.

## Waveform data

- **SDS archive:** the [`SDSArchive`](../configuration/waveforms.md#sds-archive) provider is the fastest. Copy unstructured data into an SDS archive for large searches.
- **Squirrel:** with a `persistent` collection, [Pyrocko Squirrel](../configuration/waveforms.md#pyrocko-squirrel) keeps its file index between runs.
- **Memory:** a shorter [`window_length`][qseek.search.Search.window_length] of the search needs less memory.
- **Images:** keep [`save_images`][qseek.search.Search.save_images] off unless you need the images; writing them costs time and disk space.

## Travel times

The [fast marching](../configuration/ray-tracers.md#fast-marching) ray tracer is faster than Pyrocko Cake for large numbers of stations and nodes. Pyrocko Cake caches its travel time tables in the cache directory, so later searches with the same model start faster. `qseek clear-cache` empties the cache.
