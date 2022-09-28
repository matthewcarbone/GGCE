---
title: 'GGCE: an efficient solver for lattice model Hamiltonians in the polaron limit'
tags:
  - Python
  - computational spectroscopy
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
    equal-contrib: true
    affiliation: 2
  - name: John Sous
    affiliation: 3
affiliations:
 - name: Computational Science Initiative, Brookhaven National Laboratory, Upton, New York 11973, USA
   index: 1
 - name: Department of Physics and Astronomy, University of British Columbia, Vancouver, British Columbia V6T 1Z1, Canada
   index: 2
 - name: Department of Physics, Stanford University, ...
   index: 3
date: TODO
bibliography: paper.bib
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
We present a Python software package, called the Generalized Green's Function Cluster Expansion (GGCE) for computing fundamental properties of polarons in electron-phonon Hamiltonians. The software is built on the Momentum Average family of methods and based entirely on the software used in Carbone _et al_ [@carbone2021numerically; @carbone2021bond].

# Statement of need
The electron-phonon problem is of both fundamental relevance and practical importance in materials science. Electron-phonon interactions promote a variety of novel states in low-temperature phases and high-temperature transport in quantum materials, and are essential to the understanding of the behavior of solar cells and semiconductors.  In the dilute carrier density limit, electron-phonon coupling gives rise to polarons whose properties encode the physics of applied materials all the way from low to high temperatures.  Current reasearch on polarons is broadly divided into two themes, fundamental research carried out in theoretical physics departments and applied research in applied physics and materials science departments.  The former focuses on the many-body aspects of the polaron problem, while the latter focuses on the materials and ab initio aspects of it.  But, the two cannot be separated.  This paper provides a scientific software whose goal is to bridge this gap.  This software allows the treatment of polarons statics and dynamics in (almost) arbitary models of electron-phonon coupling.  It presents a first step towards a multiyear effort to combine ab initio understanding of materials and exact many-body analysis of polaron states. The software allows the computation of both ground-state and spectral response of polaron systems described by arbitary electron density and/or hopping-phonon coupling.  We provide a detailed manual and tutorials to aid users in the use of this software package.

# Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-FG02-97ER25308. Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References
