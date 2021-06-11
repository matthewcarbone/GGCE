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
from ggce.executors.serial import SerialDenseExecutor
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS

# run PETSc benchmark at cloud size cloud_ext with up to bosons_max bosons
cloud_ext = 10
bosons_min = 3
bosons_max = 7
bosons_step = 1
bosons_array = np.arange(bosons_min, bosons_max+1, 1)

# number of iterations to generate speed error bars
iters = 5

# set system parameters
Omega = 1.25
coupling = 2.5
hopping = 0.1
eta = 0.005

# pick specific point to do a calculation for speed benchmarking
w = -3.
k = 0.5 * np.pi

# loop over the range of allowed N_bosons to execute one computation (k,w)
# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

parallel_times = np.zeros((len(bosons_array),iters))
parallel_greens = np.zeros((len(bosons_array),iters), dtype='complex')

for ii, n in enumerate(bosons_array):

    model = Model()
    model.set_parameters(hopping=hopping)
    model.add_coupling(
        "EdwardsFermionBoson", Omega=Omega, M=cloud_ext, N=n,
        dimensionless_coupling=coupling
    )

    # repeat the calculation iters times for error bar generation
    for jj in range(iters):
        executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
        executor.prime()

        result = executor.solve(k, w, eta)

        if COMM.Get_rank() == 0:
            G, meta = result
            parallel_times[ii, jj] = meta['time'][-1]
            parallel_greens[ii, jj] = G

# now use the head node to do the serial calculation
if COMM.Get_rank() == 0:

    serial_times = np.zeros((len(bosons_array),iters))
    serial_greens = np.zeros((len(bosons_array),iters), dtype='complex')
    # do the same serially on the head node
    for ii, n in enumerate(bosons_array):

        model = Model()
        model.set_parameters(hopping=hopping)
        model.add_coupling(
            "EdwardsFermionBoson", Omega=Omega, M=cloud_ext, N=n,
            dimensionless_coupling=coupling
        )

        # repeat the calculation iters times for error bar generation
        for jj in range(iters):
            executor = SerialDenseExecutor(model, "info", mpi_comm=COMM)
            executor.prime()

            G, meta = executor.solve(k, w, eta)

            serial_times[ii,jj] = meta['time'][-1]
            serial_greens[ii,jj] = G

## now we print to disk (and plot, if have matplotlib) the two timings to compare
if COMM.Get_rank() == 0:

    # first calculate averages and standard error
    time_average_serial = np.average(serial_times, axis=-1)
    greens_average_serial = np.average(serial_greens, axis=-1)
    std_error_serial = [np.std(serial_times[ii,:], ddof=1) / \
                                np.sqrt(len(serial_times[ii,:])) \
                                        for ii in range(len(bosons_array)) ]
    time_average_parallel = np.average(parallel_times, axis=-1)
    greens_average_parallel = np.average(parallel_greens, axis=-1)
    std_error_parallel = [np.std(parallel_times[ii,:], ddof=1) / \
                                np.sqrt(len(parallel_times[ii,:])) \
                                        for ii in range(len(bosons_array)) ]
    xx = np.array([time_average_serial, std_error_serial, \
                        time_average_parallel, std_error_parallel, \
                                            greens_average_serial, greens_average_parallel]).T
    np.savetxt(os.path.join(script_dir,f"speed_benchmark_M_{str(cloud_ext)}_N_{bosons_min}_{bosons_max}.txt"), xx,\
                header = f"serial time (s)    serial time error (s)    parallel time (s)    parallel time error (s)"
                f"    serial G    parallel G",\
                fmt = ['%.3f %+.0ej', '%.3f %+.0ej', '%.3f %+.0ej', '%.3f %+.0ej' ,'%.5e %+.5ej', '%.5e %+.5ej'],\
                delimiter = "    " )

    # uncomment if speed benchmark visualization desired
    plt.scatter(bosons_array, \
                time_average_serial, \
                label = f"serial, dense (cont'd frac.), M = {cloud_ext}")
    plt.scatter(bosons_array, \
                time_average_parallel, \
                label = f"parallel, sparse (PETSc), M = {cloud_ext}")
    plt.errorbar(bosons_array, time_average_serial, 0, \
                std_error_serial, 'none', color = 'b')
    plt.errorbar(bosons_array, time_average_parallel, 0, \
                std_error_parallel, 'none', color = 'r',)
    plt.xlabel('N bosons', fontsize=16)
    plt.xticks(size=12)
    plt.ylabel('CPU time, seconds', fontsize=16)
    plt.yticks(size=12)
    plt.title(f"Speed comparison for M={cloud_ext}, N=[{bosons_min} to {bosons_max}]", fontsize=18)
    plt.legend(fontsize=14)
    # plt.show()
    plt.savefig(os.path.join(script_dir,f'benchmark_petsc_M_{str(cloud_ext)}_N_{str(bosons_min)}_{str(bosons_max)}.png'),format='png', bbox_inches='tight')
