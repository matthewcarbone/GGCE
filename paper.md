---
title: 'GGCE: A software package for simulating polarons'
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
    orcid: 0000-0001-9413-6344
    affiliation: "4, 5"
  - name: Mona Berciu
    orcid: 0000-0002-6736-1893
    affiliation: "2, 3"
  - name: David R. Reichman
    orcid: 0000-0002-5265-5637
    affiliation: 6
  - name: John Sous
    corresponding: true
    orcid: 0000-0002-9994-5789
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
We present an efficient implementation of the Generalized Green's function Cluster Expansion (GGCE) approach, which is a new method for computing the ground-state properties and dynamics of polarons (single electrons coupled to lattice vibrations) in model electron-phonon systems. GGCE is well suited for a variety of electron-phonon couplings, including, but not restricted to, site and bond Holstein and Peierls (or Su-Schrieffer-Heeger) couplings and couplings to multiple phonon modes with different energy scales and coupling type. The approach can be used directly to study observables at zero temperature or in conjunction with a new formalism and new solvers to study finite-temperature observables. This program can be deployed using small scale solver as well as large scale ones like the PETSc sparse matrix solver engine.

# Statement of need
The electron-phonon problem is of both fundamental relevance and practical importance in materials science [@ephReview1; @ephReview2]. Electron-phonon interactions promote a variety of states, low-temperature phases and high-temperature transport phenomena in quantum materials, and are essential to the understanding of the behavior of solar cells [@solarcells] and semiconductors [@semiconductors]. In the dilute carrier density limit, electron-phonon coupling gives rise to quasiparticles called polarons whose properties encode the physics of materials spanning the full range of low to high temperatures. 

Research on polarons has been divided into two streams: fundamental research focussed on qualitative aspects [@Mahan] and applied research focussed on obtaining quantitative information related to specific applied physics and materials science questions. This paper presents a new scientific software that aims to bridge this gap. It allows for the treatment of polaron statics and dynamics in models of electron-phonon coupling of almost arbitrary form provided that they are sufficiently short-ranged. It presents a self-contained first step in an ongoing effort to combine _ab initio_ understanding of materials and exact many-body analysis of polaron states. The software allows for a _numerically exact_ computation of both the ground-state and spectral response of polaron systems described by arbitrary electron density- and/or hopping-phonon coupling [@carbone2021numerically].


# Software summary

Our software, the Python software package called GGCE, is designed to function as a versatile, self-generated code to solve for the properties of polarons in model electron-phonon systems. The software is based on the Generalized Green's function Cluster Expansion (GGCE), which is a numerically exact extension of a variational approach known in the theoretical physics community as the Momentum Average (MA) approach [@Berciu2006prl; @Goodvin2006prb]: details on the theoretical framework of GGCE can be found in Carbone _et al_ [@carbone2021numerically; @carbone2021bond; @carbonefuture1]. The fundamental insight of the MA approximation is to utilize a variational space formed of clouds of spatially clustered phonon configurations. Via comparison with exact methods, this approximation was shown to yield quantitatively accurate results. In order to systematically converge MA to the limit of infinite Hilbert space dimension, the cloud size, which serves as a control parameter, is taken to infinity [@Goodvin2006prb]. This, however, required derivation of the set of equations corresponding to a given cloud size for all cloud sizes smaller tha a cutoff, then repeating this for increasing values of the cuttoff until convegence is achieved. The ever increasing complexity of the structure of the system of equations at large cloud sizes means that this approach can become human intesnive very quickly especially in the regime of small phonon energies where large cloud sizes are usually needed in order to converge to the numerically exact limit [@Dominic; @Sous1; @Sous2]. Carbone _et al_ articulated a formulation these features of MA in order to present GGCE as an algorithm which automates the generation and solution of the systems of equations on the computer for arbiratry cloud size, thus completely removing the need for human intervention [@carbone2021numerically]. Benchmarks of GGCE on several model systems verified that the convergence with cloud size is fast, rendering this an eficient and controlled numerically exact method even in the most challenging limits.  

The formulation described in Ref. [@carbone2021numerically] is designed to study the zero-temperature behavior of polarons. A recent advance [@carbonefuture1; @carbonefuture2] allows us to combine this approach with the Thermofield Double formalism [@suzuki1985thermo; @takahashi1996thermo], in which the problem at finite temperature is exactly equivalent to one at zero-temperature with coupling to real and fictious phonons. Benchmarks of this approach shows that the method is competitive and perhaps superior to state-of-the-art methods like matrix-product-state methods [@DMRGpaper] especially in the limit of small phonon frequencies. 

The software package functions as an on-the-fly generator of the equations of motion for the single-particle Green's function for a given set of control parameters (cloud size and phonon number) and input model parameters (energy scales and coupling strength). The generated system of equations is then solved numerically in order to obtain the Green's function of interest using a solver chosen out of a selection available for the user. The software provides various subroutines for postprocessing of the Green's function, including methods to compute the ground state energy, polaron dispersion, quasiparticle weight and single-particle spectral functions. The user can call the zero temperature solvers or the finite temperature solvers to access the different functionalities.

The GGCE code consists of three components detailed in our documentation: models, systems and solvers.

- Models completely describe the Hamiltonian system to solve, and the level of theory (specified by the control parameters) at which to solve it;
- Systems construct all of the objects required to build the matrix to solve the system of equations;
- Solvers utilize different back-ends to actually solve the constructed matrix in an efficient manner.

## Models

The choice of model completely defines the type of electron-phonon coupling used in the Hamiltonian. Every model Hamiltonian implemented to-date assumes a lattice with nearest-neighbor hopping of electrons and Einstein (dispersionless) phonons. The user can input the electron hopping amplitude, the phonon frequency and the type and strength of the electron-phonon coupling. Currently, we have implemented the Holstein, site-Peierls (Su-Schrieffer-Heeger), bond-Peierls and Edwards fermion-boson models (as well as any arbitrary combination of these). The details of the implementation can be found at `ggce.model.py`.

## Systems

The Systems objects are a helpful intermediary for performing the sometimes expensive step of constructing the Python objects required for building matrices of linear equations. Systems are instantiated from a Model. At creation, using the information in the Model about the electron-phonon couplings, they automatically construct and store the equations-of-motion object called the "basis". The basis remains "un-evaluated": it is passed into the Solver, where it can used to obtain a matrix of equation coefficients at any values of momentum and energy. In this way, the basis is constructed only once, and then simply called repeatedly to determine the coefficients. Construction of the basis follows the scheme outlined in Carbone _et al_ [@carbone2021numerically]. For relatively small clouds, the basis can be visualized for sanity-checking the calculation, using a method that pretty-prints the equations structure.

Provided with a root directory, System will also automatically checkpoint the basis to disk using pickle, allowing for a later restart of a failed computation or for restarting long jobs on a cluster with time-limited jobs without the expensive re-computation of the basis.

## Solvers

At the heart of the GGCE engine are the solvers, which implement different approaches to solving the linear systems of equations obtained by a System object. The simplest of these use NumPy's dense solver, which solves the equation-of-motion matrix using a continued fraction approach [@Goodvin2006prb], or SciPy's sparse solver. For truly large-scale computations with sizable phonon clouds and/or many different electron-phonon couplings operating simultaneously, GGCE interfaces with the powerful, massively parallel PESTc sparse solver engine. All GGCE Solvers are MPI-enabled, and allow for a variety of parallelization schemes, all of which are detailed in our documentation. At the extreme, the PETSc interface can parallelize runs across momentum-energy points and also parallelize solving of a single large sparse matrix at each point, and thus allowing for straightforward use of all available cluster resources.

The spectrum Solver method allows the user to quickly evaluate the Green's function at a specified range of momenta and energies. Like the System, it automatically checkpoints the solution (Green's function value) at every momentum-energy point using pickle, allowing for restart in case of failure or time limits on cluster jobs.

# Acknowledgements

M. R. C. acknowledges the following support: This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-FG02-97ER25308. S. F. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Stewart Blusson Quantum Matter Institute (SBQMI). J. S. acknowledges support from the Gordon and Betty Moore Foundation’s EPiQS Initiative through Grant GBMF8686 at Stanford University.

Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References
