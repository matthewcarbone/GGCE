<div align=center>

# The Generalized Green's function Cluster Expansion

[![image](https://joss.theoj.org/papers/688705844ea344353b86815d8345f8d5/status.svg)](https://joss.theoj.org/papers/688705844ea344353b86815d8345f8d5)
[![image](https://github.com/matthewcarbone/GGCE/actions/workflows/ci.yml/badge.svg)](https://github.com/matthewcarbone/GGCE/actions/workflows/ci.yml)
[![image](https://github.com/matthewcarbone/GGCE/actions/workflows/ci_petsc.yml/badge.svg)](https://github.com/matthewcarbone/GGCE/actions/workflows/ci_petsc.yml)
[![codecov](https://codecov.io/gh/matthewcarbone/GGCE/branch/master/graph/badge.svg?token=6Q7EUWBW6O)](https://codecov.io/gh/matthewcarbone/GGCE)
[![image](https://app.codacy.com/project/badge/Grade/bdb53153835a49fa8921b28a519b2ead)](https://www.codacy.com/gh/matthewcarbone/GGCE/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=matthewcarbone/GGCE&amp;utm_campaign=Badge_Grade)
[![python](https://img.shields.io/badge/-Python_3.7+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

**Numerically exact Green's functions for lattice polaron models, easily and efficiently**

_If you use this work, please cite us!_ <br>
[GGCE Software (JOSS)](https://doi.org/10.21105/joss.05115) | [Original Paper (PRB)](https://doi.org/10.1103/PhysRevB.104.035106) | [Bond-Peierls Paper (PRB)](https://doi.org/10.1103/PhysRevB.104.L140307)

</div>
   
## 💾 Installation

Just use `pip`!

```bash
conda create -n py3.9-ggce python=3.9 -y
conda activate py3.9-ggce
pip install ggce
```

We provide detailed information on _advanced_ installation at our [documentation](https://matthewcarbone.github.io/GGCE/installation.html).

## 🚀 Quickstart

Get started in a few lines of code right [here](https://matthewcarbone.github.io/GGCE/tutorials/introduction.html)!

Or, checkout our intro tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matthewcarbone/GGCE/blob/master/notebooks/GGCE_Tutorial.ipynb)

## ⚠️ Known issues

Currently, PETSc is not building properly (we believe this is a problem with the `petsc4py` wheel or something like this).
If you would like to help contribute to this bugfix, please see this [PR](https://github.com/matthewcarbone/GGCE/pull/76)!

## 🙏 Acknowledgements

This software is based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Advanced Scientific Computing
Research, Department of Energy Computational Science Graduate Fellowship
under Award Number DE-FG02-97ER25308. This work also used theory and
computational resources of the Center for Functional Nanomaterials,
which is a U.S. Department of Energy Office of Science User Facility,
and the Scientific Data and Computing Center, a component of the
Computational Science Initiative, at Brookhaven National Laboratory
under Contract No. DE-SC0012704. We also acknowledge support from the
Natural Sciences and Engineering Research Council of Canada (NSERC) and
the Stewart Blusson Quantum Matter Institute (SBQMI).
