# before we import numpy, set the number of threads for numpy multithreading
# typically most gains are achieved by 4-6 threads
import numpy as np

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
# import matplotlib.pyplot as plt

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append(ggce_head_dir)
    import ggce

from petsc4py import PETSc

from ggce.executors.serial import SerialDenseExecutor
from ggce.executors.petsc4py.parallel import ParallelSparseExecutor

# run PETSc benchmark at cloud size cloud_ext with up to bosons_max bosons
cloud_ext = 10
bosons_max = 8

omega = 1.25
coupling = 2.5
hopping = 0.1
eta = 0.005

# pick specific point to do a calculation for speed benchmarking
w = -5.5
k = 0.5 * np.pi

# loop over the range of allowed N_bosons to execute one computation (k,w)
# start with PETSc
COMM = PETSc.COMM_WORLD

parallel_times = np.zeros(bosons_max)
parallel_greens = np.zeros(bosons_max, dtype='complex')

for n in range(1,bosons_max+1):

    system_params = {
        "model": ["EFB"],
        "M_extent": [cloud_ext],
        "N_bosons": [n],
        "Omega": [omega],
        "dimensionless_coupling": [coupling],
        "hopping": hopping,
        "broadening": eta,
        "protocol": "zero temperature"
    }

    executor = ParallelSparseExecutor(system_params, "info", mpi_comm=COMM)
    executor.prime()

    G, meta = executor.solve(k, w)
    parallel_times[n-1] = meta['time'][-1]
    parallel_greens[n-1] = G

# now use the head node to do the serial calculation
if COMM.getRank() == 0:

    serial_times = np.zeros(bosons_max)
    serial_greens = np.zeros(bosons_max, dtype='complex')
    # do the same serially on the head node
    for n in range(1,bosons_max+1):

        system_params = {
            "model": ["EFB"],
            "M_extent": [cloud_ext],
            "N_bosons": [n],
            "Omega": [omega],
            "dimensionless_coupling": [coupling],
            "hopping": hopping,
            "broadening": eta,
            "protocol": "zero temperature"
        }

        executor = SerialDenseExecutor(system_params, "info")
        executor.prime()

        G, meta = executor.solve(k, w)
        serial_times[n-1] = meta['time'][-1]
        serial_greens[n-1] = G

## now we print to disk (and plot, if have matplotlib) the two timings to compare
if COMM.getRank() == 0:

    xx = np.array([serial_times, parallel_times, serial_greens, parallel_greens]).T
    np.savetxt(os.path.join(script_dir,f"speed_benchmark_M_{str(cloud_ext)}_n_{str(bosons_max)}.txt"), xx,\
                header = f"serial time (s)    parallel time (s)    serial G    parallel G",\
                fmt = ['%.3f %+.0ej', '%.3f %+.0ej' ,'%.5e %+.5ej', '%.5e %+.5ej'],\
                delimiter = "    " )

    # uncomment if speed benchmark visualization desired
    # plt.scatter(range(1,bosons_max+1), \
    #             serial_times, \
    #             label = f"serial, dense (cont'd frac.), M = {cloud_ext}")
    # plt.scatter(range(1,bosons_max+1), \
    #             parallel_times, \
    #             label = f"parallel, sparse (PETSc), M = {cloud_ext}")
    # plt.xlabel('N bosons')
    # plt.ylabel('CPU time, seconds')
    # plt.legend()
    # plt.savefig(os.path.join(script_dir,f'benchmark_petsc_m{str(cloud_ext)}_n{str(bosons_max)}.png'),format='png')
