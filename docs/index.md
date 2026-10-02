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

<div class="grid cards qs-four qs-pillars" markdown>

-   :lucide-brain-circuit:{ .lg .middle } __Data and AI driven__

    Machine learning pickers, trained on large seismic data sets, annotate the phase arrivals. Stacking them over the whole network makes the detection robust.

    [How Qseek works :lucide-arrow-right:](concepts/how-it-works.md)

-   :lucide-workflow:{ .lg .middle } __Automatic__

    From continuous waveforms to located earthquakes with phase picks, magnitudes and features, without manual picking.

    [Quick start :lucide-arrow-right:](getting-started/quick-start.md)

-   :lucide-zap:{ .lg .middle } __Extremely fast__

    Built for large networks and years of data, with compiled stacking and phase annotation on the GPU.

    [Benchmark :lucide-arrow-right:](about/benchmark.md)

-   :lucide-globe:{ .lg .middle } __Geological settings__

    Applied to tectonic swarms, volcano-tectonic unrest and induced seismicity at geothermal sites.

    [Showcase :lucide-arrow-right:](about/showcase.md)

</div>

## Features

<div class="grid cards qs-four qs-features" markdown>

-   :lucide-activity:{ .lg .middle } __Machine learning phase detection__

    PhaseNet, EQTransformer, OBSTransformer and LFEDetect through [SeisBench](https://github.com/seisbench/seisbench), on CPU or GPU.

    [Image functions :lucide-arrow-right:](configuration/image-functions.md)

-   :lucide-boxes:{ .lg .middle } __Adaptive octree__

    The search volume refines itself around the seismic sources, for fast searches and precise locations.

    [Search volume :lucide-arrow-right:](configuration/search-volume.md)

-   :lucide-layers:{ .lg .middle } __1D and 3D velocity models__

    Constant velocity, 1D layered models with fast marching or Pyrocko Cake, and 3D NonLinLoc models.

    [Ray tracers :lucide-arrow-right:](configuration/ray-tracers.md)

-   :lucide-clock:{ .lg .middle } __Station corrections__

    Station-specific and source-specific travel time corrections, extracted from previous runs.

    [Station corrections :lucide-arrow-right:](configuration/station-corrections.md)

-   :lucide-gauge:{ .lg .middle } __Magnitudes__

    Local magnitudes (ML) with regional attenuation models and moment magnitudes (Mw) from modeled peak amplitudes.

    [Magnitudes :lucide-arrow-right:](configuration/magnitudes.md)

-   :lucide-radio-tower:{ .lg .middle } __Real-time monitoring__

    Stream waveforms from SeedLink servers and send detection alerts to Telegram.

    [Real-time monitoring :lucide-arrow-right:](guides/real-time.md)

-   :lucide-chart-scatter:{ .lg .middle } __Web UI__

    Explore detections, magnitudes and stations in the browser, also for runs on remote machines.

    [Explore results :lucide-arrow-right:](results/explore.md)

-   :lucide-file-output:{ .lg .middle } __Open formats__

    Detections as JSON, CSV and Pyrocko markers, and export to HypoDD for double-difference relocation and to VELEST for velocity model inversion.

    [Export detections :lucide-arrow-right:](guides/manage-runs.md#export-the-detections)

</div>

<div class="qs-cite" markdown>

:lucide-quote:{ .lg } __Cite Qseek__

Please cite the Qseek paper when you use it in your work.

=== "Reference"

    Isken, M., Niemz, P., Münchmeyer, J., Büyükakpınar, P., Heimann, S., Cesca, S., Vasyura-Bathke, H., & Dahm, T. (2025). Qseek: A data-driven Framework for Automated Earthquake Detection, Localization and Characterization. *Seismica*, 4(1). [doi:10.26443/seismica.v4i1.1283](https://doi.org/10.26443/seismica.v4i1.1283)

=== "BibTeX"

    ```bibtex
    @article{isken2025qseek,
      title   = {Qseek: A data-driven Framework for Automated Earthquake
                 Detection, Localization and Characterization},
      author  = {Isken, M. and Niemz, P. and M{\"u}nchmeyer, J. and
                 B{\"u}y{\"u}kakp{\i}nar, P. and Heimann, S. and Cesca, S. and
                 Vasyura-Bathke, H. and Dahm, T.},
      journal = {Seismica},
      year    = {2025},
      volume  = {4},
      number  = {1},
      doi     = {10.26443/seismica.v4i1.1283}
    }
    ```

</div>

## Supported by

<div class="qs-supporters" markdown>

[![GFZ Helmholtz Centre for Geosciences](https://www.gfz.de/fileadmin/gfz/medien_kommunikation/Infothek/Mediathek/Bilder/GFZ/GFZ_Logo/GFZ-Wortbildmarke-DE-Helmholtzdunkelblau_RGB.svg)](https://gfz.de)
[![SeisBench](https://seisbench.readthedocs.io/en/stable/_images/seisbench_logo_subtitle_outlined.svg)](https://github.com/seisbench/seisbench)
[![](https://pyrocko.org/docs/current/_images/pyrocko_shadow.png) Pyrocko](https://pyrocko.org)

</div>
