# Run with e.g.
# mpiexec -np `n_processes` python 001_Tutorial.py
# to start in parallel

import numpy as np

# uncomment if want to visualize PETSc speed benchmark
import matplotlib as mpl
import matplotlib.pyplot as plt

from petsc4py import PETSc
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

from ggce.executors.petsc4py.parallel import ParallelSparseExecutor
from ggce.executors.serial import SerialDenseExecutor

COMM = MPI.COMM_WORLD

literature_data = np.loadtxt(os.path.join(script_dir,"000_example_A.txt"))

system_params = {
    "model": ["EFB"],
    "M_extent": [4],
    "N_bosons": [9],
    "Omega": [1.25],
    "dimensionless_coupling": [2.5],
    "hopping": 0.1,
    "broadening": 0.005,
    "protocol": "zero temperature"
}

executor = ParallelSparseExecutor(system_params, "info", mpi_comm=COMM)
# executor = SerialDenseExecutor(system_params, "info")
executor.prime()

wgrid = np.linspace(-5.5, -2.5, 100)
spectrum = [executor.solve(0.5 * np.pi, w) for w in wgrid]

if COMM.Get_rank() == 0:
    spectrum = np.array([-s[0].imag / np.pi for s in spectrum])
    xx = np.array([wgrid.squeeze(), spectrum.squeeze()]).T
    np.savetxt(os.path.join(script_dir,f"parallel_results.txt"), xx)

    # Compare with "ground truth" (see https://journals.aps.org/prb/
    # abstract/10.1103/PhysRevB.82.085116, figure 5 center) via:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    M, N = system_params["M_extent"][0], system_params["N_bosons"][0]
    ax.plot(wgrid, spectrum / spectrum.max(), 'k', label = rf'PETSc ($M={M}, N={N}$)')
    ax.plot(literature_data[:, 0], literature_data[:, 1] / literature_data[:,1].max(),\
                                'r--', linewidth=0.5, label='Ground truth')
    ax.set_ylabel("$A(\pi/2, \omega)$ [normalized]")
    ax.set_xlabel("$\omega$")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig(os.path.join(script_dir,f'petsc_vs_groundtruth_M_{M}_N_{N}.png'), format='png', bbox_inches='tight')
