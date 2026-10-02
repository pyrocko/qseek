---
icon: lucide/house
hide:
  - navigation
  - toc
---

<div class="qs-hero" markdown>
<div class="qs-hero__bg" aria-hidden="true"></div>
<div class="qs-hero__text" markdown>

# Find the earthquakes hidden in your seismic data

Qseek detects and locates earthquakes in large seismic data sets. It stacks machine learning phase annotations along modeled travel times and focuses an adaptive octree on the seismic sources, in continuous archives and in real time.

[:lucide-rocket: Get started](getting-started/installation.md){ .md-button .md-button--primary }
[:lucide-settings-2: Configuration](configuration/index.md){ .md-button }

</div>
</div>

<div class="grid cards qs-four" markdown>

-   :lucide-brain-circuit:{ .lg .middle } __Data and AI driven__

    ---

    Machine learning pickers, trained on large seismic data sets, annotate the phase arrivals. Stacking them over the whole network makes the detection robust.

    [:lucide-arrow-right: How Qseek works](concepts/how-it-works.md)

-   :lucide-workflow:{ .lg .middle } __Automatic__

    ---

    From continuous waveforms to located earthquakes with phase picks, magnitudes and features, without manual picking.

    [:lucide-arrow-right: Quick start](getting-started/quick-start.md)

-   :lucide-zap:{ .lg .middle } __Extremely fast__

    ---

    Built for large networks and years of data, with compiled stacking and phase annotation on the GPU.

    [:lucide-arrow-right: Benchmark](about/benchmark.md)

-   :lucide-globe:{ .lg .middle } __Geological settings__

    ---

    Applied to tectonic swarms, volcano-tectonic unrest and induced seismicity at geothermal sites.

    [:lucide-arrow-right: Showcase](about/showcase.md)

</div>

## Features

<div class="grid cards qs-four" markdown>

-   :lucide-activity:{ .lg .middle } __Machine learning phase detection__

    ---

    PhaseNet, EQTransformer, OBSTransformer and LFEDetect through [SeisBench](https://github.com/seisbench/seisbench), on CPU or GPU.

    [:lucide-arrow-right: Image functions](configuration/image-functions.md)

-   :lucide-boxes:{ .lg .middle } __Adaptive octree__

    ---

    The search volume refines itself around the seismic sources, for fast searches and precise locations.

    [:lucide-arrow-right: Search volume](configuration/search-volume.md)

-   :lucide-layers:{ .lg .middle } __1D and 3D velocity models__

    ---

    Constant velocity, 1D layered models with fast marching or Pyrocko Cake, and 3D NonLinLoc models.

    [:lucide-arrow-right: Ray tracers](configuration/ray-tracers.md)

-   :lucide-clock:{ .lg .middle } __Station corrections__

    ---

    Station-specific and source-specific travel time corrections, extracted from previous runs.

    [:lucide-arrow-right: Station corrections](configuration/station-corrections.md)

-   :lucide-gauge:{ .lg .middle } __Magnitudes__

    ---

    Local magnitudes (ML) with regional attenuation models and moment magnitudes (Mw) from modeled peak amplitudes.

    [:lucide-arrow-right: Magnitudes](configuration/magnitudes.md)

-   :lucide-radio-tower:{ .lg .middle } __Real-time monitoring__

    ---

    Stream waveforms from SeedLink servers and send detection alerts to Telegram.

    [:lucide-arrow-right: Real-time monitoring](guides/real-time.md)

-   :lucide-chart-scatter:{ .lg .middle } __Web UI__

    ---

    Explore detections, magnitudes and stations in the browser, also for runs on remote machines.

    [:lucide-arrow-right: Explore results](results/explore.md)

-   :lucide-file-output:{ .lg .middle } __Open formats__

    ---

    Detections as JSON, CSV and Pyrocko markers, and export to VELEST for velocity model inversion.

    [:lucide-arrow-right: Export detections](guides/manage-runs.md#export-the-detections)

</div>

!!! abstract "Cite Qseek"
    Isken, M., Niemz, P., Münchmeyer, J., Büyükakpınar, P., Heimann, S., Cesca, S., Vasyura-Bathke, H., & Dahm, T. (2025). Qseek: A data-driven Framework for Automated Earthquake Detection, Localization and Characterization. Seismica, 4(1). [https://doi.org/10.26443/seismica.v4i1.1283](https://doi.org/10.26443/seismica.v4i1.1283)

## Supported by

<div class="qs-supporters" markdown>

[![GFZ](https://www.gfz.de/fileadmin/gfz/medien_kommunikation/Infothek/Mediathek/Bilder/GFZ/GFZ_Logo/GFZ-Wortbildmarke-DE-Helmholtzdunkelblau_RGB.svg){ width="240" }](https://gfz.de)
[![SeisBench](https://seisbench.readthedocs.io/en/stable/_images/seisbench_logo_subtitle_outlined.svg){ width="200" }](https://github.com/seisbench/seisbench)
[![Pyrocko](https://pyrocko.org/docs/current/_images/pyrocko_shadow.png){ width="45" }](https://pyrocko.org)

</div>
