# before we import numpy, set the number of threads for numpy multithreading
# typically most gains are achieved by 4-6 threads
import numpy as np

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append(ggce_head_dir)
    import ggce

from mpi4py import MPI

from ggce.model import Model  # noqa: E402
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS

# run PETSc benchmark at cloud size cloud_ext with up to bosons_max bosons
cloud_ext = 8
bosons_min = 10
bosons_max = 10
bosons_step = 1
bosons_array = np.arange(bosons_min, bosons_max+1, 1)

# set system parameters
Omega = 1.25
coupling = 2.5
hopping = 0.1
eta = 0.005

# pick a few points to do a calculation for parallel optimization
pts = 5
wgrid = np.linspace(-5.5, -2.5, pts)
k = 0.5 * np.pi

# loop over the range of allowed N_bosons to execute one computation (k,w)
# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

parallel_times = np.zeros((len(bosons_array),pts))
parallel_greens = np.zeros((len(bosons_array),pts), dtype='complex')

for ii, n in enumerate(bosons_array):

    model = Model()
    model.set_parameters(hopping=hopping)
    model.add_coupling(
        "EdwardsFermionBoson", Omega=Omega, M=cloud_ext, N=n,
        dimensionless_coupling=coupling
    )

    # repeat the calculation iters times for error bar generation
    for jj, w in enumerate(wgrid):
        executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
        executor.prime()

        result = executor.solve(k, w, eta)

        if COMM.Get_rank() == 0:
            G, meta = result
            parallel_times[ii, jj] = meta['time'][-1]
            parallel_greens[ii, jj] = G

## now we print to disk
if COMM.Get_rank() == 0:

    # first calculate averages and standard error
    time_average_parallel = np.average(parallel_times, axis=-1)
    std_error_parallel = [np.std(parallel_times[ii,:], ddof=1) / \
                                np.sqrt(len(parallel_times[ii,:])) \
                                        for ii in range(len(bosons_array)) ]

    xx = np.array([time_average_parallel, std_error_parallel]).T
    np.savetxt(os.path.join(script_dir,f"speed_benchmark_M_{str(cloud_ext)}_N_{bosons_min}_{bosons_max}.txt"), xx,\
                header = f"parallel time (s)    parallel time error (s)",\
                fmt = '%.3f',\
                delimiter = "    " )
