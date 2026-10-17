---
icon: lucide/cpu
---

# Benchmark

Qseek is built for large-N data sets: many stations and long time spans. A 600 GB data set, about 700 years of waveforms, takes about two days on a 64-core machine with one Nvidia A100 GPU.

| Stations | Data throughput | Waveform time processed per second |
| --- | --- | --- |
| 300+ | 50 MB/s | 12 hours |
| 50 | 200 MB/s | 6 hours |

!!! note
    The throughput depends on the resolution of the octree and on the number of detected events: every detection refines the octree and adds picks, magnitudes and features.

## Why Qseek is fast

- **Concurrent processing:** Qseek loads, pre-processes and annotates the next waveforms while it stacks the current ones, with Python [asyncio](https://docs.python.org/3/library/asyncio.html) and threads.
- **Compiled stacking:** the delay-and-sum stacking and migration runs in C extensions, parallelized with OpenMP.
- **GPU annotation:** the machine learning phase annotation runs on the GPU with [PyTorch](https://pytorch.org/).
- **Adaptive octree:** only the nodes around detected events are refined, see [octree refinement](../concepts/how-it-works.md#octree-refinement).

The [performance guide](../guides/performance.md) shows which settings make your search faster.

## Station weights

The [station weights](../configuration/station-weights.md) decide which events the stack detects and where it locates them. Their defaults are chosen on the examples of the [playground](../getting-started/playground.md).

### Playground

Each example searched with `MADTrigger`, `mad_factor` 10, and the defaults of each station weights model. A detection is well constrained with at least 8 picks, at Norcia with at least 20:

| Example | Station weights | Detections | Well constrained | Residual RMS, median | Catalog events matched | Epicenter offset, median |
| --- | --- | --- | --- | --- | --- | --- |
| Campi Flegrei, 1 day, 18 stations | `DistanceWeights` | 731 | 521 | 0.285 s | 45 / 45 | 242 m |
| | `StationDensityWeights` | 753 | 512 | 0.281 s | 45 / 45 | 247 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 750 | 523 | 0.284 s | 45 / 45 | 240 m |
| | `DensityTaperWeights` | 727 | 523 | 0.293 s | 45 / 45 | 244 m |
| | `DensityPlateauWeights` | 708 | 514 | 0.288 s | 45 / 45 | 238 m |
| Campi Flegrei, 10 days, 19 stations | `DistanceWeights` | 6382 | 3760 | 0.278 s | 209 / 211 | 277 m |
| | `StationDensityWeights` | 6320 | 3711 | 0.279 s | 209 / 211 | 263 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 6623 | 3772 | 0.279 s | 209 / 211 | 288 m |
| | `DensityTaperWeights` | 6329 | 3754 | 0.281 s | 209 / 211 | 280 m |
| | `DensityPlateauWeights` | 6021 | 3697 | 0.287 s | 209 / 211 | 273 m |
| Mount Spurr, 3 days, 10 local and 10 regional stations | `DistanceWeights` | 1756 | 1048 | 0.346 s | 228 / 234 | 419 m |
| | `StationDensityWeights` | 1774 | 1051 | 0.348 s | 228 / 234 | 440 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 2214 | 1074 | 0.320 s | 230 / 234 | 360 m |
| | `DensityTaperWeights` | 1770 | 1051 | 0.346 s | 228 / 234 | 390 m |
| | `DensityPlateauWeights` | 1656 | 1030 | 0.357 s | 227 / 234 | 449 m |
| Norcia, 3 days, 49 stations with a cluster of 12 | `DistanceWeights` | 3734 | 3603 | 0.309 s | 1013 / 1071 | 644 m |
| | `StationDensityWeights` | 4036 | 3817 | 0.312 s | 1024 / 1071 | 646 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 3971 | 3746 | 0.313 s | 1020 / 1071 | 651 m |
| | `DensityTaperWeights` | 3841 | 3673 | 0.311 s | 1018 / 1071 | 645 m |
| | `DensityPlateauWeights` | 3726 | 3609 | 0.309 s | 1016 / 1071 | 654 m |

The residual RMS grows with the number of picks, and the catalogs list only the larger events with locations of their own. At Norcia, the INGV catalog lists routine locations, 4 km deeper than the relocations of the sequence (Chiaraluce et al., 2017). Neither measures how consistently the detections are located.

### Relative locations

HypoDD relocates the detections of the `DistanceWeights` search with catalog and cross-correlation differential times. The double-difference residuals of the cross-correlation data, evaluated at the locations of each search, measure how consistently a search locates neighboring events, independent of its picks. Lower is better. Campi Flegrei, 1 day, 340 detections in all searches; Norcia, 3574 detections:

| Station weights | Campi Flegrei, median | Campi Flegrei, 90th percentile | Norcia, median | Norcia, 90th percentile |
| --- | --- | --- | --- | --- |
| `DistanceWeights` | 53.6 ms | 168 ms | 42.6 ms | 153 ms |
| `StationDensityWeights` | 57.6 ms | 187 ms | 41.5 ms | 151 ms |
| `DistanceWeights`, `"nearest_neighbor"` | 54.0 ms | 173 ms | 41.3 ms | 152 ms |
| `DensityTaperWeights` | 53.9 ms | 169 ms | 42.2 ms | 152 ms |
| `DensityPlateauWeights` | 56.4 ms | 188 ms | 42.8 ms | 154 ms |
| `StationDensityWeights`, `taper_stations` 8, `taper_cap` 1.0 | 56.7 ms | 194 ms | 42.7 ms | 167 ms |
| `null`, all stations the same weight | 56.9 ms | 190 ms | | |

The rows of `DensityTaperWeights` and `DensityPlateauWeights` come from evaluations with 340 and 339 detections on Campi Flegrei and 3598 and 3555 at Norcia, in which `DistanceWeights` reach 168 ms and 153 ms (90th percentile). `DensityPlateauWeights` with 2 and 3 independent stations on the plateau reach 185 ms and 182 ms on Campi Flegrei.

On Campi Flegrei, `DistanceWeights` with 6 and 8 stations of full weight reach 174 ms and 183 ms (90th percentile), with 2 and 3 stations 175 ms and 172 ms, with a taper of 6 km and 16 km instead of the 11 km of `"mean_interstation"` 188 ms and 177 ms. The evaluation favors the search that HypoDD relocated: relocated from `DistanceWeights` with 8 stations of full weight instead, `DistanceWeights` with 4, 6 and 8 stations reach 178 ms, 181 ms and 171 ms, and all other station weights stay above them.

A short taper locates the detections least consistently. With `taper_stations` 8 and `taper_cap` 1.0, the 10-day search on Campi Flegrei places detections on walls up to 3 km deep. At Norcia, 14% of the well-constrained detections lie above sea level, 4.5% with `DistanceWeights`, and 382 residuals at their locations exceed the output format of HypoDD. The weights are normalized per node, so a taper that changes steeply between neighboring nodes moves the maximum of the stack. Without any taper, the noise of distant stations degrades the locations as well.
