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
## Glitch rise model (`deepska.glitch`)

The pulsar glitch rise model of
[Graber, Cumming and Andersson 2018, ApJ 865, 23](https://doi.org/10.3847/1538-4357/aad776)
written in JAX. It builds the crust from the Negele and Vautherin 1973 equation of state
with the TOV equations, splits the pinned superfluid into cylindrical shells, gets the
mutual friction from the Epstein and Baym pinning cases A, B and C, and then runs the spin
equations (Eqs 20 to 22 of the paper) forward in time. Everything is float64 and can be
vmapped, so sweeps over the inputs are quick.

Three scripts, each with its settings at the top of the file:

```sh
python -m deepska.glitch.scripts.paper        # Table 2 and the 13 figures of the paper
python -m deepska.glitch.scripts.sweep        # sweep any inputs and compare with the Vela data
python -m deepska.glitch.scripts.sensitivity  # how much each output moves with each input
```

There is one function per file, so a change to one piece stays in that file. Names are the
paper's symbols in lowercase (`b_core`, `omega0`, `i_sf`, `r_drip`), and inputs that always
go together are grouped in `deepska/glitch/kinds.py` (`StarInputs`, `Spin`, `Star`). A
single run is `param_to_output(star_inputs, b_profile, b_core, spin)`.

Autodiff through the star build (`m0`, `r0`, `rho_d`, `rho_cc`, `rho_min`) should not be
trusted. The surface is found by a straight line across the last TOV row, so the radius is
a small sawtooth in those inputs and the gradient comes out as the slope of one tooth. The
sensitivity script uses a 1% nudge each way for those inputs and autodiff for the rest.

The Vela 2016 glitch timing residuals in
`deepska/glitch/data/palfreyman2018_vela_glitch_residuals.csv` are the supplementary data
file of [Palfreyman et al. 2018, Nature 556, 219](https://doi.org/10.1038/s41586-018-0001-x)
(`41586_2018_1_MOESM1_ESM.csv`, the same file that is in
[vanessagraber/glitchrises](https://github.com/vanessagraber/glitchrises)). It is only used
to compare the model with the data.