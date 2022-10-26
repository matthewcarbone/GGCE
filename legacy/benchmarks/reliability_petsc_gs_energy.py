############ Stress-testing PETSc ##########
"""
In this script we use PETSc reproduce some features of Fig. 2 in arxiv 2103.01972.
This is the final check that PETSc is reliable.
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
from ggce.executors.serial import SerialDenseExecutor

# run convergence for cloud size up to cloud_max with up to bosons_max bosons
cloud_min = 3
cloud_max = 3
bosons_min = 30
bosons_max = 30

# fix parameter values same as ground truth from the https://export.arxiv.org/pdf/2103.01972
# article
Omega = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01])
coupling = np.linspace(0.0, 1.2, 30)
hopping = 1.0
eta = 0.005
next_lam_offset = 0.1
w0 = -2.1  # initial guess
model_type = "Holstein"

# define the k grid for the dispersion calculation
# since just looking for GS of Holstein, that is always at k = 0
kgrid = np.array([0])

# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

# start by creating a folder with the text and image output in run directory
# only head node make directories
results_dir = os.path.join(script_dir, "relb_gs_rlts")

if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

# create a list to store gs energies
gs_energies = np.zeros(
    (
        cloud_max - cloud_min + 1,
        bosons_max - bosons_min + 1,
        len(Omega),
        len(coupling),
    )
)

# start the loop over all allowed M, N
fig, ax = plt.subplots(1, 1, figsize=(3, 2))
for m in range(cloud_min, cloud_max + 1):
    for n in range(bosons_min, bosons_max + 1):
        for ii, omega in enumerate(Omega):

            for jj, lam in enumerate(coupling):

                # initialize the model with pre-determined parameters
                model = Model()
                model.set_parameters(hopping=hopping)
                model.add_coupling(
                    model_type,
                    Omega=omega,
                    M=m,
                    N=n,
                    dimensionless_coupling=lam,
                )

                # initialize the Executor to manage the calculation
                executor = ParallelSparseExecutorMUMPS(
                    model, "info", mpi_comm=COMM
                )
                executor.prime()

                # re-define w0 so we are not wasting time
                if jj > 0:
                    w0 = (
                        gs_energies[m - cloud_min, n - bosons_min, ii, jj - 1]
                        - next_lam_offset
                    )

                # find the energy of the GS (polaron)
                results = executor.dispersion(kgrid, w0, eta, nmax=10000)
                gs_energy = results[0]["ground_state"]
                # add the gs energy to the matrix
                gs_energies[m - cloud_min, n - bosons_min, ii, jj] = gs_energy

            # # only collect and output results on the head node
            if COMM.Get_rank() == 0:
                # process the solver output to extract time to solve and spectrum vals
                ax.plot(
                    coupling,
                    gs_energies[m - cloud_min, n - bosons_min, ii, :],
                    label=rf"PETSc ($M={m}, N={n}, \Omega = {omega}$)",
                )
                ax.set_ylim(-3.1, -1.9)
                ax.set_xlim(0, 1.2)
                ax.set_ylabel(r"$E_{GS} / t$", fontsize=16)
                ax.set_xlabel(r"$\lambda$", fontsize=16)

                # also output the energy data to disk for posterity / postprocess
                xx = np.array(
                    [
                        coupling,
                        gs_energies[m - cloud_min, n - bosons_min, ii, :],
                    ]
                ).T
                np.savetxt(
                    os.path.join(
                        results_dir,
                        f"gs_energy_vs_lam_M_{m}_N_{n}_om_{omega}.txt",
                    ),
                    xx,
                    header=f"lambda        E_GS for Omega = {omega} and (M,N) = ({m},{n})",
                    delimiter="    ",
                )


if COMM.Get_rank() == 0:
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.show()
    plt.savefig(
        os.path.join(
            results_dir, f"gs_energy_vs_lam_M_{cloud_max}_N_{bosons_max}.png"
        ),
        format="png",
        bbox_inches="tight",
    )
    # also output the energy data to disk for posterity / postprocess
    xx = np.array(
        [
            coupling,
        ]
        + [
            gs_energies[m, n, l, :]
            for l in range(gs_energies.shape[2])
            for m in range(gs_energies.shape[0])
            for n in range(gs_energies.shape[1])
        ]
    ).T
    np.savetxt(
        os.path.join(
            results_dir,
            f"gs_energy_vs_lam_Mmax_{cloud_max}_Nmax_{bosons_max}.txt",
        ),
        xx,
        header=f"lambda        E_GS for various Omega and (M,N) arranged as Omega[0], M = 1, N = 1,..., M = 2, N = 1,... Omega[1]...",
        delimiter="    ",
    )