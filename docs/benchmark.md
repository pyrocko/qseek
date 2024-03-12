# Benchmark

## Computation Performance

Qseek is built for searching in large-N data sets. The implementation is leveraging Python [`asyncio`](https://docs.python.org/3/library/asyncio.html) heavily to implement threading and keeping the CPU busy. It is built on top of highly performant Pyrocko functions implemented in C language. The inference is using [PyTorch](https://pytorch.org/) which enables GPU computation of the seismic imaging functions.

This enables high throughput of seismic data in different scenarios.

| Number Stations | Throughput in MB | Throughput in Waveform data |
| --------------- | ---------------- | ----------------------------|
| 300+            | 50 MB/sec        | 12 hours/sec                |
| 50              | 200 MB/sec        |  6 hours/sec                |

Scanning a 600 GB (~700 years of waveforms) data set costs **~2 days on a 64 cores machine equipped with an Nvidia A100 GPU**.

!!! note
    The performance depends heavily on the octree resolution and the number of events detected in the data set.

## Related Projects

A list of other projects using stacking and migration approach to back-project seismic energy sources in 3D space:

### Lassie-v1

Lassie - The friendly Earthquake detector in version 1. Qseek utilizes the same optimized heavy-duty functions for stacking and migration as Lassie v1.

[Lassie-v1 on Pyrocko Git](https://git.pyrocko.org/pyrocko/lassie)

### QuakeMigrate

QuakeMigrate uses a waveform migration and stacking algorithm to search for coherent seismic phase arrivals across a network of instruments. It produces—from raw data—catalogues of earthquakes with locations, origin times, phase arrival picks, and local magnitude estimates, as well as rigorous estimates of the associated uncertainties.

[QuakeMigrate on GitHub](https://github.com/QuakeMigrate/QuakeMigrate)

### BPMF

Complete framework for earthquake detection and location: Backprojection and matched-filtering (BPMF), with methods for automatic picking, relocation and efficient waveform stacking.

[BPMF on GitHub](https://github.com/ebeauce/Seismic_BPMF)

### Loki

LOKI (LOcation of seismic events through traveltime staKIng) is a code that performs earthquake detection and location using waveform coherence analysis (waveform stacking).

[Loki on GitHub](https://github.com/wulwife/LOKI)

### MALMI

MALMI (MAchine Learning aided earthquake MIgration location), variant of Loki for detecting and locating earthquakes using ML image functions provided by SeisBench.

[MALMI on GitHub](https://github.com/speedshi/MALMI)
