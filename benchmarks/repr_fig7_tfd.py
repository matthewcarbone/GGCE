# before we import numpy, set the number of threads for numpy multithreading
# typically most gains are achieved by 4-6 threads
import numpy as np

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import shutil

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

_CLEAN_START_ = True

# run PETSc benchmark at cloud size cloud_ext with up to bosons_max bosons
cloud_ext_H = 3
bosons_H = 9

cloud_ext_tfd_H = 3
bosons_tfd_H = 5

Omega_H = 1.
coupling_H = 1.
temperature = 0.4 * Omega_H
hopping = 1.
eta = 0.005
model_type1 = "Holstein"

# define the k grid for the dispersion calculation
kgrid = np.array([0])
# pick several points to generate error bars
wgrid = np.linspace(-6., 6., 10)

# loop over the range of allowed N_bosons to execute one computation (k,w)
# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

results_dir = os.path.join(script_dir, 'repr_fig7_dmrg')

if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if os.path.exists(results_dir):
        if _CLEAN_START_:
            shutil.rmtree(results_dir)
            os.mkdir(results_dir)
        else:
            print(f"if you want a clean start, be careful!")

model = Model("repr_fig7_dmrg", "info", log_file=None)
model.set_parameters(hopping=hopping, temperature = temperature, \
            lattice_constant=1.0, dimension=1, max_bosons_per_site=None)
model.add_coupling(
    model_type1, Omega=Omega_H, M=cloud_ext_H, N=bosons_H, \
    M_tfd = cloud_ext_tfd_H, N_tfd = bosons_tfd_H,
    dimensionless_coupling=coupling_H)
model.finalize()

# print model parameters for the log
if COMM.Get_rank() == 0:
    model.visualize()

executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
executor.prime()

for ii, k in enumerate(kgrid):
    for jj, w in enumerate(wgrid):

        g_func, exitcodes_dict = executor.solve(k, w, eta)
        a_func = - g_func.imag / np.pi
        ## now we print to disk (and plot, if have matplotlib) the two timings to compare
        if COMM.Get_rank() == 0:

            times = exitcodes_dict['time'][0]
            mumps_exit_codes = exitcodes_dict['mumps_exit_code'][0]
            mumps_mem_tot = exitcodes_dict['mumps_mem_tot'][0]
            tol_excess = exitcodes_dict['manual_tolerance_excess'][0]

            # also output the spectrum data to disk for posterity / postprocess
            # create first entries which are momentum
            xx = np.array([[k, w, a_func, times, mumps_exit_codes, \
                                            mumps_mem_tot, tol_excess]])
            # create the output file
            # if this is the first time then create the file with the heading
            result_file = os.path.join(results_dir, f"repr_fig7_dmrg.out")
            if not os.path.exists(result_file):
                np.savetxt(result_file, xx,\
                            header = f"momentum                    omega                         "
                            f"spectral func               time_to_compute (sec)    "
                            f"MUMPS convergence code (0 = good)  MUMPS total memory used (Gb)  "
                            f"Tolerance excess  (|r| - rtol*|b|)",\
                            delimiter = '    ')
            else:
                with open(result_file, "a") as datafile:
                    # datafile.write("\n")
                    np.savetxt(datafile, xx, delimiter = '    ')

## plot the spectrum function cut for fig. 7 comparison
if COMM.Get_rank() == 0:
    result_file = os.path.join(results_dir, f"repr_fig7_dmrg.out")
    ## now let's visualize the spectrum and see what we have
    k_vals, w_vals, a_func, times, mumps_exit_codes, mumps_mem_tot, tol_excess = \
                                np.loadtxt(result_file, unpack = True, skiprows=1)
    # here it will be only a single k-slice
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(w_vals, a_func, 'b', label = rf'GGCE ($M={cloud_ext_H}, N={bosons_H}$)')
    ax.set_ylabel(f"$A(0, \omega)$", size = 16)
    ax.set_xlabel("$\omega / \omega_0$", size = 16)
    ax.tick_params(labelsize=14)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([0, 1, 2])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 2.5)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,f'repr_fig7_dmrg_M_{cloud_ext_H}_N_{bosons_H}.png'), format='png')
