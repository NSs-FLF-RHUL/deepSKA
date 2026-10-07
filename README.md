# deepSKA

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/tests.yml
[linting-badge]:            https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/NSs-FLF-RHUL/deepska/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-GPLv3-blue.svg
<!-- prettier-ignore-end -->

## About

:warning: This package is currently under construction and in pre-release.
The API and features may change suddenly without warning.

### Project Team

Vanessa Graber ([vanessa.graber@rhul.ac.uk](mailto:vanessa.graber@rhul.ac.uk))

<!-- TODO: how do we have an array of collaborators - steal from s2fft -->

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`deepska` requires Python 3.11.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `deepska` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/NSs-FLF-RHUL/deepska.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/NSs-FLF-RHUL/deepska.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```
