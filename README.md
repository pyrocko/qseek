# Lassie

*The friendly earthquake detector*

[![Build and test](https://github.com/miili/lassie-v2/actions/workflows/build.yaml/badge.svg)](https://github.com/miili/lassie-v2/actions/workflows/build.yaml)
![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<!-- [![PyPI](https://img.shields.io/pypi/v/lassie)](https://pypi.org/project/lassie/) -->

Lassie is an earthquake detection and localisation framework based on stacking and migration method. It combines neural network phase picks with an iterative octree localisation approach for accurate localisation of seismic events.

Key features are of the earthquake detection and localisation framework are:

* Earthquake phase detection using machine-learning pickers from [SeisBench](https://github.com/seisbench/seisbench)
* Octree localisation approach for efficient and accurate search
* Different velocity models:
  * Constant velocity
  * 1D Layered velocity model
  * 3D fast-marching velocity model (NonLinLoc compatible)
* Extraction of earthquake event features:
    * Local magnitudes
    * Ground motion attributes
* Automatic extraction of modelled and picked travel times
* Calculation and application of station corrections / station delay times

Lassie is built on top of [Pyrocko](https://pyrocko.org).

## Installation

```sh
git clone https://github.com/pyrocko/lassie-v2
cd lassie-v2
pip3 install .
```

## Project Initialisation

Initialize a new project in a fresh directory.

```sh
lassie init my-project/
```

Edit the `my-project.json`

Start the earthquake detection with

```sh
lassie run search.json
```

## Packaging

The simplest and recommended way of installing from source:

### Development

Local development through pip.

```sh
cd lassie-v2
pip3 install .[dev]
```

The project utilizes pre-commit for clean commits, install the hooks via:

```sh
pre-commit install
```

## Citation

Please cite lassie as:

> TBD

## License

Contribution and merge requests by the community are welcome!

Lassie-v2 was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
