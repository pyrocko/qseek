# Qseek

*The friendly earthquake detector*

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Build and test](https://github.com/pyrocko/qseek/actions/workflows/build.yaml/badge.svg)](https://github.com/pyrocko/qseek/actions/workflows/build.yaml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
[![Documentation](https://img.shields.io/badge/read-documentation-blue)](https://pyrocko.github.io/qseek/)
<!-- [![PyPI](https://img.shields.io/pypi/v/lassie)](https://pypi.org/project/lassie/) -->

Qseek is a data-driven earthquake detection and localisation framework for large seismic data sets. The framework is based on a stacking and migration approach, a beamforming method. It combines neural network phase annotations with an iterative octree localisation approach for efficient and accurate localisation of seismic events.

Key features are of the earthquake detection and localisation framework are:

* Earthquake phase detection using machine-learning model from [SeisBench](https://github.com/seisbench/seisbench), pre-trained on different data sets.
  * [PhaseNet (Zhu and Beroza, 2018](https://doi.org/10.1093/gji/ggy423)
  * [EQTransformer (Mousavi et al., 2020)](https://doi.org/10.1038/s41467-020-17591-w)
  * [OBSTransformer (Niksejel and Zahng, 2024)](https://doi.org/10.1093/gji/ggae049)
  * LFEDetect
* Octree localisation approach for efficient and accurate search
* Different velocity models:
  * Constant velocity
  * 1D Layered velocity model
  * 3D fast-marching velocity model (NonLinLoc compatible)
* Extraction of earthquake event features:
  * Local magnitudes (ML), different attenuation models
  * Moment Magnitudes (MW) based on modelled peak ground motions
  * Different ground motion attributes (e.g. PGA, PGV, ...)
* Automatic extraction of modelled and picked travel times
* Station Corrections
  * station specific corrections (SST)
  * source specific station corrections (SSST)

Qseek is built on top of [Pyrocko](https://pyrocko.org).

## Documentation

For more information check out the documentation at <https://pyrocko.github.io/qseek/>.

## Installation

Simple installation from GitHub.

```sh
pip install git+https://github.com/pyrocko/qseek
```

## Project Initialisation

Show the default config.

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

> Marius Paul Isken, Peter Niemz, Jannes Münchmeyer, Sebastian Heimann, Simone Cesca, Torsten Dahm, Qseek: A data-driven Framework for Machine-Learning Earthquake Detection, Localization and Characterization, Seismica, 2024, *submitted*

## License

Contribution and merge requests by the community are welcome!

Qseek was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
