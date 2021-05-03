# Run with e.g.
# mpirun -n `n_processes` python 001_Tutorial.py
# to start in parallel

import numpy as np

# import matplotlib as mpl
# import matplotlib.pyplot as plt

from petsc4py import PETSc

import sys

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append("..")
    import ggce

from ggce.executors.parallel import ParallelSparseExecutor
# from ggce.executors.parallel import SerialSparseExecutor

literature_data = np.loadtxt("000_example_A.txt")

system_params = {
    "model": ["EFB"],
    "M_extent": [3], # 5
    "N_bosons": [9], #10
    "Omega": [1.25],
    "dimensionless_coupling": [2.5],
    "hopping": 0.1,
    "broadening": 0.005,
    "protocol": "zero temperature"
}

COMM = PETSc.COMM_WORLD

executor = ParallelSparseExecutor(system_params, "DEBUG", mpi_comm=COMM)
executor.prime()

wgrid = np.linspace(-5.5, -2.5, 100)
spectrum = [executor.solve(0.5 * np.pi, w) for w in wgrid]

if COMM.rank == 0:
    spectrum = np.array([-s[0].imag / np.pi for s in spectrum])
    xx = np.array([wgrid.squeeze(), spectrum.squeeze()]).T
    np.savetxt("parallel_results.txt", xx)


# fig, ax = plt.subplots(1, 1, figsize=(3, 2))
# ax.plot(wgrid, spectrum / spectrum.max(), 'k')
# ax.plot(literature_data[:, 0], literature_data[:, 1], 'r--', linewidth=0.5)
# ax.set_ylabel("$A(\pi/2, \omega) / \max A(\pi/2, \omega)$")
# ax.set_xlabel("$\omega$")
# plt.savefig('test.png', format='png')
