#!/usr/bin/env python3

# Run with e.g.
# mpiexec -np `n_processes` python 002_PETSc_simple_example.y
# to start in parallel

import numpy as np

# uncomment if want to visualize PETSc accuracy benchmark
# import matplotlib as mpl
# import matplotlib.pyplot as plt

from mpi4py import MPI

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
head_ggce_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append(head_ggce_dir)
    import ggce

# Import the parallel executor and load in the MPI communicator
from ggce.model import Model  # noqa: E402
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS  # noqa: E402

COMM = MPI.COMM_WORLD

# load the "grouth truth" data from https://journals.aps.org/prb/
# abstract/10.1103/PhysRevB.82.085116, figure 5 center for comparison
literature_data = np.loadtxt(os.path.join(script_dir,"000_example_A.txt"))

# Select the input parameters and load in the parallel model
M, N = 3, 9
model = Model()
model.set_parameters(hopping=0.1)
model.add_coupling(
    "EdwardsFermionBoson", Omega=1.25, M=M, N=N,
    dimensionless_coupling=2.5
)

# Set up an executor to manage the calculation
executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
executor.prime()

# arrange a omega array and use list comprehension to execute calculations
wgrid = np.linspace(-5.5, -2.5, 100)
spectrum = [executor.solve(0.5 * np.pi, w, 0.005) for w in wgrid]

# Results are returned on RANK 0 only
if COMM.Get_rank() == 0:
    spectrum = np.array([-s[0].imag / np.pi for s in spectrum])
    xx = np.array([wgrid.squeeze(), spectrum.squeeze()]).T
    np.savetxt(os.path.join(script_dir,f"parallel_results.txt"), xx)

    # Compare with "ground truth" (see https://journals.aps.org/prb/
    # abstract/10.1103/PhysRevB.82.085116, figure 5 center) via:
    # fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    # ax.plot(wgrid, spectrum / spectrum.max(), 'k', label = rf'PETSc ($M={M}, N={N}$)')
    # ax.plot(literature_data[:, 0], literature_data[:, 1] / literature_data[:,1].max(),\
    #                             'r--', linewidth=0.5, label='Ground truth')
    # ax.set_ylabel("$A(\pi/2, \omega)$ [normalized]")
    # ax.set_xlabel("$\omega$")
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    # plt.savefig(os.path.join(script_dir,f'petsc_mumps_vs_groundtruth_M_{M}_N_{N}.png'), format='png', bbox_inches='tight')
