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
