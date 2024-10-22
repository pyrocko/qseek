# Qseek

*The friendly earthquake detector*

[![Build and test](https://github.com/pyrocko/qseek/actions/workflows/build.yaml/badge.svg)](https://github.com/pyrocko/qseek/actions/workflows/build.yaml)
[![Documentation](https://img.shields.io/badge/read-documentation-blue)](https://pyrocko.github.io/qseek/)
![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<!-- [![PyPI](https://img.shields.io/pypi/v/lassie)](https://pypi.org/project/lassie/) -->

Qseek is an earthquake detection and localisation framework based on stacking and migration, a beamforming method. It combines neural network phase picks with an iterative octree localisation approach for accurate localisation of seismic events.

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
  * Moment Magnitudes (MW) based on modelled peak ground motions
  * Local magnitudes (ML), different models
  * Ground motion attributes (e.g. PGA, PGV, ...)
* Automatic extraction of modelled and picked travel times
* Station Corrections
  * station specific corrections (SST)
  * source specific station corrections (SSST)

Qseek is built on top of [Pyrocko](https://pyrocko.org).

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
