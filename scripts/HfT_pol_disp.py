############ Stress-testing PETSc ##########
'''
In this script we use PETSc to calculate the dispersion of the polaron.
'''
########### ################

# before we import numpy, set the number of threads for numpy multithreading
# using the OMP_NUM_THREADS environment variable
import numpy as np
from scipy.optimize import curve_fit

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import shutil
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"\
                f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expan"\
                f"sion/new_lorentzfit"
sys.path.append(ggce_head_dir)
import ggce

# load the mpi communicator
from mpi4py import MPI

# load the Model and Executor classes to store parameters and manage the calculation
from ggce.model import Model  # noqa: E402
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS
from ggce.executors.serial import SerialDenseExecutor
from ggce.utils.utils import lorentzian

M, N, M2, N2 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

a, b, c = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
w_init = float(sys.argv[8])

# run PETSc benchmark at cloud size cloud_ext with up to bosons_max bosons
cloud_ext_H = M
bosons_H = N

cloud_ext_tfd_H = M2
bosons_tfd_H = N2

coupling_H = a
Omega_H = b
temperature = c * Omega_H
hopping = 1.
model_type1 = "Holstein"

# define parameters for the dispersion calculation
kgrid = np.linspace(0, np.pi/20., 20)
eta = 0.05
next_k_offset_factor = 1.
eta_step_div = 10.

# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

results_dir = os.path.join(script_dir, f'pol_dis_HfT_{M}{N}_{M2}{N2}')

_CLEAN_START_ = True
if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if os.path.exists(results_dir):
        if _CLEAN_START_:
            shutil.rmtree(results_dir)
            os.mkdir(results_dir)
        else:
            print(f"if you want a clean start, be careful!")

model = Model("finite_t_calc", "info", log_file=None)
model.set_parameters(hopping=hopping, temperature = temperature, \
            lattice_constant=1.0, dimension=1, max_bosons_per_site=None)
model.add_coupling(
    model_type1, Omega=Omega_H, M=cloud_ext_H, N=bosons_H, \
    M_tfd = cloud_ext_tfd_H, N_tfd = bosons_tfd_H,
    dimensionless_coupling=coupling_H)
model.finalize()

# initialize the Executor to manage the calculation
executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
# executor = SerialDenseExecutor(model, "info", mpi_comm=COMM)
executor.prime()

results = executor.dispersion(kgrid, w_init, eta, nmax = 10000, \
                eta_step_div = eta_step_div, next_k_offset_factor = next_k_offset_factor, peak_routine = "change_w")
## this is a dict with keys k, w, A, ground_state (which is w of highest point) and weight (quasiparticle)
## so I should be fitting to pairs of (k, ground_state)
energy_k = [results[ii]['ground_state'] for ii in range(len(kgrid))]

# # only collect and output results on the head node
if COMM.Get_rank() == 0:

    # plot and save the Lorentz fit for diagnostic purposes
    for ii in range(len(kgrid)):
        fig, ax = plt.subplots(1, 1)
        loc = results[ii]["ground_state"]
        scale = results[ii]["weight"]
        ax.plot(results[ii]["w"], lorentzian(results[ii]["w"], loc, scale, eta), label = "Lorentz fit")
        ax.scatter(results[ii]["w"], results[ii]["A"], label = "Calculation")
        ax.set_ylabel(r"$A(k,w)$", fontsize=20)
        ax.set_xlabel(r"$w$", fontsize=20)
        ax.tick_params(which = 'both', labelsize = 16)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,f'lorentzfit_k_{kgrid[ii]:.3f}.png'),\
                        format='png')
        plt.close(fig)

    # process the solver output to extract time to solve and spectrum vals
    fig, ax = plt.subplots(1, 1)
    ax.scatter(kgrid, energy_k, label = 'PETSc')
    ax.set_ylabel(r"$E_P(k)$", fontsize=20)
    ax.set_xlabel(r"$k/\pi$", fontsize=20)
    ax.tick_params(which = "both", labelsize = 16)

    def quadratic_band(k, m_star, offset):

        return k**2 / (2.*m_star) - offset

    best_fit_pars, covmat = curve_fit(quadratic_band, kgrid, energy_k, \
                                                    p0 = [0.5, energy_k[0]])
    dense_kgrid = np.linspace(0, kgrid[-1], 100)
    ax.plot( dense_kgrid, quadratic_band(dense_kgrid, *best_fit_pars), \
                                                        label = f"$k^2$ fit")

    # also output the energy data to disk for posterity / postprocess
    xx = np.array([kgrid, energy_k]).T
    np.savetxt(os.path.join(results_dir,f"polaron_dispersion.txt"), xx,\
                header = f"momentum        E_P(k) with m_star / m_0 = {best_fit_pars[0] / 0.5041061511616247} (m_0 = 0.5041061511616247)",\
                delimiter = '    ')

    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results_dir,f'polaron_dispersion.png'), format='png', bbox_inches='tight')

    ## also save all data just in case
    for ii, kval in enumerate(kgrid):
        xx = np.array([results[ii]['w'], results[ii]['A']]).T
        np.savetxt(os.path.join(results_dir,f"all_results_k_{kval:.2f}.txt"), xx,\
                    header = f"omega            spectral function        for momentum = {kval}",\
                    delimiter = '    ')
