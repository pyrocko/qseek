# Qseek

*The wholesome earthquake detector*

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Build and test](https://github.com/pyrocko/qseek/actions/workflows/build.yaml/badge.svg)](https://github.com/pyrocko/qseek/actions/workflows/build.yaml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
![PyPI - Version](https://img.shields.io/pypi/v/qseek)
[![Documentation](https://img.shields.io/badge/read-documentation-blue)](https://pyrocko.github.io/qseek/)
<!-- [![PyPI](https://img.shields.io/pypi/v/lassie)](https://pypi.org/project/lassie/) -->

Qseek is a data-driven earthquake detection and localisation framework driven by machine-learning and designed for large seismic data sets. The method is based on a stacking and migration approach, a beamforming method. It combines neural network phase annotations with an iterative octree localisation approach for accurate localisation of seismicity.

Key features are of the earthquake detection and localisation framework are:

* Earthquake phase detection using machine-learning model from [SeisBench](https://github.com/seisbench/seisbench), pre-trained on different data sets.
  * [PhaseNet (Zhu and Beroza, 2018)](https://doi.org/10.1093/gji/ggy423)
  * [EQTransformer (Mousavi et al., 2020)](https://doi.org/10.1038/s41467-020-17591-w)
  * [OBSTransformer (Niksejel and Zahng, 2024)](https://doi.org/10.1093/gji/ggae049)
  * LFEDetect
* Octree localisation for efficient and accurate search
* Different velocity models:
  * Constant velocity
  * 1D Layered velocity model
  * 3D fast-marching velocity model (NonLinLoc compatible)
* Earthquake magnitudes and other features:
  * Local magnitudes (ML) with different attenuation models
  * Moment Magnitudes (MW) based on modelled attenuation curves ([Dahm et al., 2024](https://doi.org/10.26443/seismica.v3i2.1205))
  * Different ground motion attributes (e.g. PGA, PGV, ...)
* Station Corrections
  * station specific corrections (SST)
  * source specific station corrections (SSST)

Qseek is built on top of [Pyrocko](https://pyrocko.org).

## Documentation

An online is available at <https://pyrocko.github.io/qseek/>.

## Installation

Simple installation from GitHub.

```sh
pip install git+https://github.com/pyrocko/qseek
```

## Project Initialisation

Print the default config.

```sh
qseek config
```

Edit the `my-project.json`

Start the earthquake detection with

```sh
qseek search search.json
```

## Packaging

The simplest and recommended way of installing from source:

### Development

Local development through pip.

```sh
cd qseek
pip3 install .[dev]
```

The project utilizes pre-commit for clean commits, install the hooks via:

```sh
pre-commit install
```

## Citation

Please cite Qseek as:

> Isken, M., Niemz, P., Münchmeyer, J., Büyükakpınar, P., Heimann, S., Cesca, S., Vasyura-Bathke, H., & Dahm, T. (2025). Qseek: A data-driven Framework for Automated Earthquake Detection, Localization and Characterization. Seismica, 4(1). <https://doi.org/10.26443/seismica.v4i1.1283>

## License

Contribution and merge requests by the community are welcome!

Qseek was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
