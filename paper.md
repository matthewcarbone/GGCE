---
title: 'GGCE: A software package for studying polarons and their dynamics'
tags:
  - Python
  - computational quantum physics
  - condensed matter physics
  - materials science
authors:
  - name: Matthew R. Carbone
    orcid: 0000-0002-5181-9513
    equal-contrib: true
    corresponding: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    email: mcarbone@bnl.gov
  - name: Stepan Fomichev
    orcid: 0000-0002-1622-9382
    equal-contrib: true
    affiliation: "2, 3"
  - name: Andrew J. Millis
    affiliation: "4, 5" # (Multiple affiliations must be quoted)
  - name: Mona Berciu
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
  - name: David R. Reichman
    affiliation: 6 # (Multiple affiliations must be quoted)
  - name: John Sous
    corresponding: true
    affiliation: "7, 8"
affiliations:
 - name: Computational Science Initiative, Brookhaven National Laboratory, Upton, New York 11973, USA
   index: 1
 - name: Department of Physics and Astronomy, University of British Columbia, Vancouver, British Columbia V6T 1Z1, Canada
   index: 2
 - name: Stewart Blusson Quantum Matter Institute, University of British Columbia, Vancouver, British Columbia, V6T 1Z4 Canada
   index: 3
 - name: Department of Physics, Columbia University, New York, New York 10027, USA
   index: 4
 - name: Center for Computational Quantum Physics, Flatiron Institute, New York, New York 10010, USA
   index: 5
 - name: Department of Chemistry, Columbia University, New York, New York 10027, USA
   index: 6
 - name: Department of Physics, Stanford University, Stanford, California 93405, USA
   index: 7
 - name: Geballe Laboratory for Advanced Materials, Stanford University, Stanford, California 94305, USA
   index: 8
date: TODO
bibliography: paper.bib
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
We present an efficient implementation of the Generalized Green's function Cluster Expansion (GGCE), a new method for computing the properties of electron-phonon Hamiltonians in the single-carrier (polaron) limit. GGCE can handle almost any electron-phonon coupling linear in phonon displacement, including Holstein, Peierls and multi-phonon mode couplings. In addition, it is capable of finite-temperature calculations and can be deployed at scale using the PETSc sparse matrix solver engine.

# Statement of need
The electron-phonon problem is of both fundamental relevance and practical importance in materials science [@ephReview1; @ephReview2]. Electron-phonon interactions promote a variety of novel states, low-temperature phases and high-temperature transport phenomena in quantum materials, and are essential to the understanding of the behavior of solar cells [@solarcells] and semiconductors [@semiconductors]. In the dilute carrier density limit, electron-phonon coupling gives rise to quasiparticles called polarons, whose properties encode the physics of materials spanning the full range of low to high temperatures. Current research on polarons is broadly divided into two streams: fundamental research carried out in theoretical physics departments, and applied research in applied physics and materials science departments. The former focuses on the many-body aspects of the polaron problem [@Mahan; @MA], while the latter focuses on materials and _ab initio_ calculations [@Giustino]. The two, however, cannot be separated.

This paper summarizes a new scientific software that bridges this gap. It allows for the treatment of polaron statics and dynamics in models of electron-phonon coupling of almost arbitrary form. It presents a self-contained first step in an ongoing effort to combine _ab initio_ understanding of materials and exact many-body analysis of polaron states. The software allows for a _numerically exact_ computation of both the ground-state and spectral response of polaron systems described by arbitrary electron density and/or hopping phonon coupling [@carbone2021numerically].


# Software summary

Our software, the Python software package called ???? (INSERT-SEXY-NAME-HERE), is designed to compute fundamental properties of polarons in electron-phonon Hamiltonians. The software is built on the variational Momentum Average family of methods [@Berciu2006prl; @Goodvin2006prb]: details on the theoretical framework of GGCE can be found in Carbone _et al_ [@carbone2021numerically; @carbone2021bond]. The fundamental insight from the Momentum Average approximation that is used in GGCE is a clever guess at the nature of the polaronic state: namely, it is assumed that the electron-phonon configurations most important to polaron formation are those with a spatially localized phonon cloud. The cloud can then be variationally increased in extent and number of phonons until desired convergence is achieved.

The primary objective of GGCE is to solve for the Green's function of the polaronic system. Once the Green's function is solved for, other quantities can be easily derived with minimal additional computation. The Green's function is calculated via continued application of the Dyson identity, generating a hierarchy of auxiliary equations of motion which are truncated at some level of theory. For the first time, GGCE provides a unified, automated approach to generating this hierarchy of equations. The GGCE level of theory is completely controlled by two parameters: the maximum number of phonons allowed in the system, and the maximum extent of the phonon cloud.

The GGCE code consists of three components detailed in our documentation: models, systems and solvers.

- Models completely describe the Hamiltonian system to solve, and the level of theory at which to solve it;
- Systems construct all of the objects required to build the matrix to solve -- the "system-of-equations";
- Solvers utilize different back-ends to actually solve the constructed matrix in an efficient manner.

## Models

The choice of model completely defines the type of electron-phonon coupling used in the Hamiltonian. Every model Hamiltonian implemented to-date assumes nearest-neighbor hopping of the electron, and Einstein phonons (phonon dispersion is constant). The user can control the electron hopping strength and the phonon frequency and the type and strength of electron-phonon coupling. Currently, we have implemented the Holstein, Peierls, Bond-Peierls and Edwards Fermion Boson models (as well as an arbitrary combination of these). The details of the implementation can be found at `ggce.model.py`.

## Systems

The Systems objects are a helpful intermediary for performing the sometimes expensive step of constructing the Python objects required for building matrices. Being instantiated from a Model, at creation they automatically compute and store the generalized and exact form of the equations of motion. Provided with a root directory, they will also automatically checkpoint these to disk.

## Solvers

At the heart of the GGCE engine are the solvers, which implement different approaches to solving the linear systems of equations obtained by a System object. The simplest of these use NumPy's dense solver or SciPy's sparse solver, while for truly large-scale computations with sizable phonon clouds and/or many different electron-phonon coupling operating simultaneously, GGCE interfaces with the powerful, massively parallel PESTc sparse solver engine. Even the non-PETSc version of GGCE is MPI enabled, and allows for a variety of parallelization schemes, all of which are detailed in our documentation.

# Acknowledgements

M. R. C. acknowledges the following support: This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-FG02-97ER25308. S.F. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Stewart Blusson Quantum Matter Institute (SBQMI).  J. S. acknowledges support from the Gordon and Betty Moore Foundation’s EPiQS Initiative through Grant GBMF8686 at Stanford University.

Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References
