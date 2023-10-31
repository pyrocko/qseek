# Welcome to Lassie 🐕‍🦺

Lassie is an earthquake detection and localisation framework. It combines modern **machine learning phase detection and robust migration and stacking techniques**.

The detector is leveraging [Pyrocko](https://pyrocko.org) and [SeisBench](https://github.com/seisbench/seisbench), it is highly-performant and can search massive data sets for seismic activity efficiently.

!!! abstract "Citation"
    TDB

![Reykjanes detections](images/reykjanes-demo.webp)

*Seismic swarm activity at Iceland, Reykjanes Peninsula during a 2020 unrest. 15,000+ earthquakes detected, outlining a dike intrusion, preceeding the 2021 Fagradasfjall eruption. Visualized in [Pyrocko Sparrow](https://pyrocko.org).*

## Features

* [x] Earthquake phase detection using machine-learning pickers from [SeisBench](https://github.com/seisbench/seisbench)
* [x] Octree localisation approach for efficient and accurate search
* [x] Different velocity models:
    * [x] Constant velocity
    * [x] 1D Layered velocity model
    * [x] 3D fast-marching velocity model (NonLinLoc compatible)
* [x] Extraction of earthquake event features:
    * [x] Local magnitudes
    * [x] Ground motion attributes
* [x] Automatic extraction of modelled and picked travel times
* [x] Calculation and application of station corrections / station delay times
* [ ] Real-time analytics on streaming data (e.g. SeedLink)


[Get Started!](getting_started.md){ .md-button }

## Build with

![Pyrocko](https://pyrocko.org/docs/current/_images/pyrocko_shadow.png){ width="100" }
![SeisBench](https://seisbench.readthedocs.io/en/stable/_images/seisbench_logo_subtitle_outlined.svg){ width="400" padding-right="40" }
![GFZ](https://www.gfz-potsdam.de/fileadmin/gfz/GFZ.svg){ width="100" }
