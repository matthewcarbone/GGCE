############ Stress-testing PETSc ##########
"""
In this script we push PETSc to the limit. We will calculate entire spectral
slices, tracking both time to evaluate as well as saving the actual spectra.
We are looking to evalaute convergence wiht (M,N), as well as time to calculate
"""
########### ################

# before we import numpy, set the number of threads for numpy multithreading
# using the OMP_NUM_THREADS environment variable
import numpy as np

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ggce
except ModuleNotFoundError:
    sys.path.append(ggce_head_dir)
    import ggce

# load the mpi communicator
from mpi4py import MPI

# load the Model and Executor classes to store parameters and manage the calculation
from ggce.model import Model  # noqa: E402
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS

# run PETSc benchmark for cloud size up to cloud_max with up to bosons_max bosons
cloud_ext = 8
bosons_max = 10

Omega = 2.5
coupling = 1.0
hopping = 1.0
eta = 0.005

# define w and k grid for the calculation
wgrid = np.linspace(-4.0, -2.0, 5)
k = 0.0

# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

# start by creating a folder with the text and image output in run directory
# only head node make directories
results_dir = os.path.join(script_dir, "check_mem_eff")
spectra_vis_path = os.path.join(results_dir, "spectra_visuals")
text_data_path = os.path.join(results_dir, "text_data")

if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # now create nested directories for text and image respectively
    if not os.path.exists(spectra_vis_path):
        os.mkdir(spectra_vis_path)
    if not os.path.exists(text_data_path):
        os.mkdir(text_data_path)

# initialize the model with pre-determined parameters
model = Model()
model.set_parameters(hopping=hopping)
model.add_coupling(
    "Holstein",
    Omega=Omega,
    M=cloud_ext,
    N=bosons_max,
    dimensionless_coupling=coupling,
)

# initialize the Executor to manage the calculation
executor = ParallelSparseExecutorMUMPS(model, "debug", mpi_comm=COMM)
executor.prime()

# execute the solve
results = [executor.solve(k, w, eta=eta) for w in wgrid]

# only collect and output results on the head node
if COMM.Get_rank() == 0:

    # process the solver output to extract time to solve and spectrum vals
    greens_func, exitcodes_dict = np.array(results).T
    spectrum = np.array([-s.imag / np.pi for s in greens_func])
    times = np.array([exitcodes["time"][0] for exitcodes in exitcodes_dict])
    mumps_conv_codes = np.array(
        [exitcodes["mumps_exit_code"][0] for exitcodes in exitcodes_dict]
    )
    mumps_mem_tot = np.array(
        [exitcodes["mumps_mem_tot"][0] for exitcodes in exitcodes_dict]
    )
    tol_excess = np.array(
        [
            exitcodes["manual_tolerance_excess"][0]
            for exitcodes in exitcodes_dict
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(
        wgrid,
        spectrum / spectrum.max(),
        "k",
        label=rf"PETSc ($M={cloud_ext}, N={bosons_max}$)",
    )
    ax.set_ylabel(f"$A(0, \omega)$ [normalized]")
    ax.set_xlabel("$\omega$")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            spectra_vis_path, f"scope_M_{cloud_ext}_N_{bosons_max}.png"
        ),
        format="png",
    )

    # also output the spectrum data to disk for posterity / postprocess
    xx = np.array(
        [wgrid, spectrum, times, mumps_conv_codes, mumps_mem_tot, tol_excess]
    ).T
    np.savetxt(
        os.path.join(
            text_data_path,
            f"parallel_results_M_{cloud_ext}_N_{bosons_max}.txt",
        ),
        xx,
        header=f"omega                         spectral func               time_to_compute (sec)    "
        f"MUMPS convergence code (0 = good)  MUMPS total memory used (Gb)  "
        f"Tolerance excess  (|r| - rtol*|b|)",
        delimiter="    ",
    )
