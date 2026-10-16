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

The [station weights](../configuration/station-weights.md) decide which events the stack detects. Their defaults are calibrated on the examples of the [playground](../getting-started/playground.md) and on a synthetic benchmark of seven networks.

### Playground

Each example searched with the defaults of the station weights, `MADTrigger` with `mad_factor` 10. A detection is well constrained with at least 8 picks, at Norcia with at least 20:

| Example | Station weights | Detections | Well constrained | Residual RMS, median | Catalog events matched | Epicenter offset, median |
| --- | --- | --- | --- | --- | --- | --- |
| Campi Flegrei, 1 day, 18 stations | `StationDensityWeights` | 954 | 524 | 0.282 s | 45 / 45 | 216 m |
| | `DistanceWeights` | 732 | 521 | 0.284 s | 45 / 45 | 241 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 750 | 525 | 0.286 s | 45 / 45 | 238 m |
| Campi Flegrei, 10 days, 19 stations | `StationDensityWeights` | 8189 | 3799 | 0.278 s | 210 / 211 | 295 m |
| | `DistanceWeights` | 6379 | 3750 | 0.278 s | 209 / 211 | 280 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 6614 | 3772 | 0.280 s | 209 / 211 | 293 m |
| Mount Spurr, 3 days, 10 local and 10 regional stations | `StationDensityWeights` | 2457 | 1140 | 0.338 s | 229 / 234 | 409 m |
| | `DistanceWeights` | 1758 | 1046 | 0.349 s | 228 / 234 | 416 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 2210 | 1070 | 0.318 s | 230 / 234 | 348 m |
| Norcia, 3 days, 49 stations with a cluster of 12 | `StationDensityWeights` | 4279 | 3874 | 0.325 s | 1026 / 1071 | 668 m |
| | `DistanceWeights` | 3732 | 3602 | 0.310 s | 1013 / 1071 | 635 m |
| | `DistanceWeights`, `"nearest_neighbor"` | 3972 | 3747 | 0.314 s | 1020 / 1071 | 648 m |

At Mount Spurr, the `DistanceWeights` taper over twice the mean interstation distance, 166 km, keeps the regional stations 80 km to 130 km away at weights of 0.25 to 0.65. `StationDensityWeights` lower them to 0.09 and less, find 40% more detections and 94 more well-constrained detections. At Norcia, the INGV catalog lists routine locations, 4 km deeper than the relocations of the sequence (Chiaraluce et al., 2017); its epicenter offsets differ by less than 35 m between the station weights.

The cap of the taper, `taper_cap` 1.0, finds 12% and 14% more detections on Campi Flegrei (1 and 10 days) and 20% more at Mount Spurr than no cap, with 59 more well-constrained detections at Mount Spurr. A cap of 0.75 raises the residuals on Campi Flegrei, caps of 1.25 and 1.5 move the epicenters at Mount Spurr farther from the catalog. At Norcia, without a gap between the stations, the cap finds 14 fewer detections and 46 fewer well-constrained detections than no cap.

### Synthetic networks

The synthetic benchmark stacks 6 hours of phase confidences for each of seven networks, two random realizations each: an observatory cluster with co-located sensors and a regional ring; a regional network with a dense sub-array; the station layouts of Campi Flegrei, Mount Spurr and Norcia; a small network of 7 stations; and a nodal line of 40 stations 30 m apart next to 8 permanent stations. Arrivals of synthetic events, Gutenberg-Richter magnitudes, have confidences from their signal-to-noise ratio and travel times through a heterogeneous velocity model; noisy stations add false peaks. The table gives the share of the events with at least 6 picks that each station weights detect at 2 false detections per hour, and the median change of the location error against `DistanceWeights`, paired by event:

| Station weights | Observatory | Regional | Campi Flegrei | Mount Spurr | Norcia | Small | Nodal | Location change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `StationDensityWeights` | 0.86 | 0.38 | 1.00 | 0.86 | 0.34 | 0.74 | 0.69 | −3 m to 5 m |
| `DistanceWeights` | 0.65 | 0.23 | 1.00 | 0.76 | 0.22 | 0.78 | 0.84 | 0 m |
| `DistanceWeights`, `"nearest_neighbor"` | 0.60 | 0.38 | 1.00 | 0.78 | 0.27 | 0.76 | 0.52 | −2 m to 1450 m |

The station weights hardly change the location of an event once it is detected; they change which events the stack detects. On the nodal network, the median distance between neighboring sites is the 60 m spacing of the nodes. It narrows the density kernel of `StationDensityWeights`, and the `"nearest_neighbor"` taper of `DistanceWeights` leaves only the nodal line in the stack, with location errors 1.4 km larger (median).
