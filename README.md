# Lassie

*The friendly earthquake detector*

[![Build and test](https://github.com/miili/lassie-v2/actions/workflows/build.yaml/badge.svg)](https://github.com/miili/lassie-v2/actions/workflows/build.yaml)
![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<!-- [![PyPI](https://img.shields.io/pypi/v/lassie)](https://pypi.org/project/lassie/) -->

Lassie is an earthquake detector based on stacking and migration method. It combines neural network phase picks with an iterative octree localisation approach.

Key features are of the tools are:

* Phase detection using SeisBench
* Efficient and accurate Octree localisation approach
* Extraction of event features
    * Local magnitudes
    * Ground motion attributes
* Determination of station corrections

## Installation

```sh
git clone https://github.com/miili/lassie-v2
cd lassie-v2
pip3 install .
```

## Project Initialisation

Initialize a new project in a fresh directory.

```sh
lassie new project-dir/
```

Edit the `search.json`

Start the detection

```sh
lassie run search.json
```

## Packaging

The simplest and recommended way of installing from source:

### Development

Local development through pip.

```sh
cd lightguide
pip3 install .[dev]
```

The project utilizes pre-commit for clean commits, install the hooks via:

```sh
pip install pre-commit
pre-commit install
```

## Citation

Please cite lassie as:

> TBD

## License

Contribution and merge requests by the community are welcome!

Lassie-v@ was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
