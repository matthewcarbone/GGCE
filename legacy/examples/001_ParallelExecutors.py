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
from ggce.model import Model  # noqa: E402
from ggce.executors.parallel import ParallelDenseExecutor  # noqa: E402

COMM = MPI.COMM_WORLD

literature_data = np.loadtxt(os.path.join(script_dir, "000_example_A.txt"))

# Select the input parameters and load in the parallel model
model = Model()
model.set_parameters(hopping=0.1)
model.add_coupling(
    "EdwardsFermionBoson", Omega=1.25, M=3, N=9, dimensionless_coupling=2.5
)

executor = ParallelDenseExecutor(model, "info", mpi_comm=COMM)
executor.prime()

wgrid = np.linspace(-5.5, -2.5, 101)
spectrum = executor.spectrum(0.5 * np.pi, wgrid, eta=0.005)

# Results are returned on RANK 0 only
if COMM.Get_rank() == 0:
    xx = np.array([wgrid.squeeze(), spectrum.squeeze()]).T
    np.savetxt(os.path.join(script_dir, "tmp_parallel_results.txt"), xx)
