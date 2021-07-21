############ Stress-testing PETSc ##########
'''
In this script we push PETSc to the limit. We will calculate entire spectral
slices, tracking both time to evaluate as well as saving the actual spectra.
We are looking to evalaute convergence wiht (M,N), as well as time to calculate
'''
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
cloud_ext = 3
bosons_max = 9

# fix parameter values same as ground truth from the https://journals.aps.org/prb/
# abstract/10.1103/PhysRevB.82.085116, figure 5 center for comparison
Omega = 1.25
coupling = 2.5
hopping = 0.1
eta = 0.005

# load the "grouth truth" data from https://journals.aps.org/prb/
# abstract/10.1103/PhysRevB.82.085116, figure 5 center for comparison
literature_data = np.loadtxt(os.path.join(ggce_head_dir, 'examples',"000_example_A.txt"))

# define w and k grid for the calculation
wgrid = np.linspace(-5.5, -2.5, 100)
k = 0.5 * np.pi

# loop over the range of allowed N_bosons to full spectrum slice (k,wgrid)
# start with defining an MPI communicator
COMM = MPI.COMM_WORLD

# start by creating a folder with the text and image output in run directory
# only head node make directories
results_dir = os.path.join(script_dir, 'check_spectrum_code')
spectra_vis_path = os.path.join(script_dir, 'check_spectrum_code', \
                                                        'spectra_visuals')
text_data_path = os.path.join(script_dir, 'check_spectrum_code', \
                                                        'text_data')

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
    "EdwardsFermionBoson", Omega=Omega, M=cloud_ext, N=bosons_max,
    dimensionless_coupling=coupling
)

# initialize the Executor to manage the calculation
executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
executor.prime()

# execute the solve
results, exitcodes_dict = executor.spectrum(k, wgrid, eta = eta, return_meta = True)

# only collect and output results on the head node
if COMM.Get_rank() == 0:

    # process the solver output to extract time to solve and spectrum vals
    results = np.array(results)
    times = np.array([[exitcodes['time'][0] for exitcodes in exitcodes_dict_array]\
                            for exitcodes_dict_array in exitcodes_dict])
    mumps_conv_codes = np.array([[exitcodes['mumps_exit_code'][0] for exitcodes in exitcodes_dict_array]\
                            for exitcodes_dict_array in exitcodes_dict])
    mumps_mem_tot = np.array([[exitcodes['mumps_mem_tot'][0] for exitcodes in exitcodes_dict_array]\
                            for exitcodes_dict_array in exitcodes_dict])
    tol_excess = np.array([[exitcodes['manual_tolerance_excess'][0] for exitcodes in exitcodes_dict_array]\
                            for exitcodes_dict_array in exitcodes_dict])
    # Plot to compare with "ground truth" (see https://journals.aps.org/prb/
    # abstract/10.1103/PhysRevB.82.085116, figure 5 center) via:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(wgrid, results[0] / results[0].max(), 'k', label = rf'PETSc ($M={cloud_ext}, N={bosons_max}$)')
    ax.plot(literature_data[:, 0], literature_data[:, 1] / literature_data[:,1].max(),\
                                'r--', linewidth=0.5, label='Ground truth')
    ax.set_ylabel("$A(\pi/2, \omega)$ [normalized]")
    ax.set_xlabel("$\omega$")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(spectra_vis_path,f'petsc_mumps_vs_groundtruth_M_{str(m)}_N_{str(n)}.png'), format='png', bbox_inches='tight')

    # also output the spectrum data to disk for posterity / postprocess
    # xx = np.array([wgrid, spectrum, times, mumps_conv_codes, \
                                    # mumps_mem_tot, tol_excess]).T
    # np.savetxt(os.path.join(text_data_path,f"parallel_results_M_{str(m)}_N_{str(n)}.txt"), xx,\
                # header = f"omega                         spectral func               time_to_compute (sec)    "
                # f"MUMPS convergence code (0 = good)  MUMPS total memory used (Gb)  "
                # f"Tolerance excess  (|r| - rtol*|b|)",\
                # delimiter = '    ')
