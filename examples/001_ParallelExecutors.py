#!/usr/bin/env python3

# Run with e.g.
# mpiexec -np `n_processes` python 001_ParallelExecutors.py
# to start in parallel

import numpy as np

from mpi4py import MPI

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
head_ggce_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append(head_ggce_dir)
    import ggce  # noqa: F401


# Import the parallel executor and load in the MPI communicator
from ggce.executors.parallel import ParallelDenseExecutor
COMM = MPI.COMM_WORLD

literature_data = np.loadtxt(os.path.join(script_dir, "000_example_A.txt"))

# Select the input parameters and load in the parallel model
system_params = {
    "model": ["EFB"],
    "M_extent": [3],
    "N_bosons": [9],
    "Omega": [1.25],
    "dimensionless_coupling": [2.5],
    "hopping": 0.1,
    "protocol": "zero temperature"
}
executor = ParallelDenseExecutor(system_params, "info", mpi_comm=COMM)
executor.prime()

wgrid = np.linspace(-5.5, -2.5, 101)
spectrum = executor.spectrum(0.5 * np.pi, wgrid, eta=0.005)

# Results are returned on RANK 0 only
if COMM.Get_rank() == 0:
    xx = np.array([wgrid.squeeze(), spectrum.squeeze()]).T
    np.savetxt(os.path.join(script_dir, "tmp_parallel_results.txt"), xx)
