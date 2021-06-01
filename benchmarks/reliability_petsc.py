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
cloud_min = 3
cloud_max = 5
bosons_min = 6
bosons_max = 6

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
results_dir = os.path.join(script_dir, 'stress_test_results')
spectra_vis_path = os.path.join(script_dir, 'stress_test_results', \
                                                        'spectra_visuals')
text_data_path = os.path.join(script_dir, 'stress_test_results', \
                                                        'text_data')

if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # now create nested directories for text and image respectively
    if not os.path.exists(spectra_vis_path):
        os.mkdir(spectra_vis_path)
    if not os.path.exists(text_data_path):
        os.mkdir(text_data_path)

# create an array to store time to finish the calculation
parallel_times = np.zeros((bosons_max,cloud_max))

# start the loop over all allowed M, N
for m in range(cloud_min,cloud_max+1):
    for n in range(bosons_min,bosons_max+1):

        # initialize the model with pre-determined parameters
        model = Model()
        model.set_parameters(hopping=hopping)
        model.add_coupling(
            "EdwardsFermionBoson", Omega=Omega, M=m, N=n,
            dimensionless_coupling=coupling
        )

        # initialize the Executor to manage the calculation
        executor = ParallelSparseExecutorMUMPS(model, "info", mpi_comm=COMM)
        executor.prime()

        # execute the solve + time it
        t0 = time.time()
        results = [executor.solve(k, w, eta) for w in wgrid]
        dt = time.time() - t0
        # add the calculation time to the matrix
        parallel_times[n-1,m-1] = dt

        # only collect and output results on the head node
        if COMM.Get_rank() == 0:
            # process the solver output to extract time to solve and spectrum vals
            greens_func, exitcodes_dict = np.array(results).T
            spectrum = np.array([ -s.imag / np.pi for s in greens_func ])
            times = np.array([exitcodes['time'][0] for exitcodes in exitcodes_dict])
            mumps_conv_codes = np.array([exitcodes['mumps_exit_code'][0] for exitcodes in exitcodes_dict])
            mumps_mem_tot = np.array([exitcodes['mumps_mem_tot'][0] for exitcodes in exitcodes_dict])
            tol_excess = np.array([exitcodes['manual_tolerance_excess'][0] for exitcodes in exitcodes_dict])

            # Plot to compare with "ground truth" (see https://journals.aps.org/prb/
            # abstract/10.1103/PhysRevB.82.085116, figure 5 center) via:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            ax.plot(wgrid, spectrum / spectrum.max(), 'k', label = rf'PETSc ($M={m}, N={n}$)')
            ax.plot(literature_data[:, 0], literature_data[:, 1] / literature_data[:,1].max(),\
                                        'r--', linewidth=0.5, label='Ground truth')
            ax.set_ylabel("$A(\pi/2, \omega)$ [normalized]")
            ax.set_xlabel("$\omega$")
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            # plt.show()
            plt.savefig(os.path.join(spectra_vis_path,f'petsc_mumps_vs_groundtruth_M_{str(m)}_N_{str(n)}.png'), format='png', bbox_inches='tight')

            # also output the spectrum data to disk for posterity / postprocess
            xx = np.array([wgrid, spectrum, times, mumps_conv_codes, \
                                            mumps_mem_tot, tol_excess]).T
            np.savetxt(os.path.join(text_data_path,f"parallel_results_M_{str(m)}_N_{str(n)}.txt"), xx,\
                        header = f"omega                         spectral func               time_to_compute (sec)    "
                        f"MUMPS convergence code (0 = good)  MUMPS total memory used (Gb)  "
                        f"Tolerance excess  (|r| - rtol*|b|)",\
                        delimiter = '    ')


## now we print to disk the execution time for all combinations of M,N
if COMM.Get_rank() == 0:
    xx = parallel_times
    np.savetxt(os.path.join(results_dir,f"time_for_spectrum_M_{cloud_min}_{cloud_max}_N_{bosons_min}_{bosons_max}.txt"), xx,\
                header = f"parallel time (cloud size columns, boson number rows )",\
                fmt = '%.3f')
