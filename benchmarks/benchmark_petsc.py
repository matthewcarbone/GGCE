# before we import numpy, set the number of threads for numpy multithreading
# typically most gains are achieved by 4-6 threads
import os

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import petsc4py
from petsc4py import PETSc

import sys

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append("..")
    import ggce

from ggce.executors.serial import SerialDenseExecutor
from ggce.executors.parallel import ParallelSparseExecutor

cloud_max = 10 # 8
boson_max = 6 # 12
omega = 1.25
coupling = 2.5
hopping = 0.1
eta = 0.005
w = -5.5
k = 0.5 * np.pi

# we will loop over the range of allowed N_bosons to execute one computation (k,w)
# start with PETSc
comm = PETSc.COMM_WORLD
parallel_times = np.zeros(boson_max)
for n in range(1,boson_max+1):

    system_params = {
        "model": ["EFB"],
        "M_extent": [cloud_max],
        "N_bosons": [n],
        "Omega": [omega],
        "dimensionless_coupling": [coupling],
        "hopping": hopping,
        "broadening": eta,
        "protocol": "zero temperature"
    }

    executor = ParallelSparseExecutor(system_params, "INFO")
    executor.prime(comm)

    try:
        G, time_dict = executor.solve(k, w) #'''use spectrum func'''
    except TypeError:
        continue
    # print(f"this is time dict parallel {time_dict}")
    parallel_times[n-1] = time_dict['time'][-1]

# now use the head node to do the serial calculation
if comm.getRank() == 0:

    serial_times = np.zeros(boson_max)
    # do the same serially on the head node
    for n in range(1,boson_max+1):

        system_params = {
            "model": ["EFB"],
            "M_extent": [cloud_max],
            "N_bosons": [n],
            "Omega": [omega],
            "dimensionless_coupling": [coupling],
            "hopping": hopping,
            "broadening": eta,
            "protocol": "zero temperature"
        }

        executor = SerialDenseExecutor(system_params, "INFO")
        executor.prime()

        G, time_dict = executor.solve(k, w) #'''use spectrum func'''
        # print(f"this is time dict serial {time_dict}")
        serial_times[n-1] = time_dict['time'][-1]

## now we plot the two timings to compare
if comm.getRank() == 0:
    plt.scatter(range(1,boson_max+1), \
                serial_times, \
                label = f"serial, dense (cont'd frac.), M = {cloud_max}")
    plt.scatter(range(1,boson_max+1), \
                parallel_times, \
                label = f"parallel, sparse (PETSc), M = {cloud_max}")
    plt.xlabel('N bosons')
    plt.ylabel('CPU time, seconds')
    plt.legend()
    plt.savefig(f'benchmark_petsc_m{str(cloud_max)}_n{str(boson_max)}.png',format='png')
