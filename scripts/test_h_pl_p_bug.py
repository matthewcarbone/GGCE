import numpy as np
# from matplotlib import rc
import matplotlib.pyplot as plt
# rc('text.latex', preamble=r'\usepackage{cmbright}')
# rc('text', usetex=True)
import sys
import os
import shutil
import time
from datetime import date

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"\
                f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expansion/"\
                f"fixing_h_pl_p_bug"
server_name = "local"
jobID = "pleasefillmein"

sys.path.append(ggce_head_dir)
import ggce

from mpi4py import MPI

from ggce.model import Model  # noqa: E402
from ggce.executors.serial import SerialDenseExecutor
from ggce.executors.serial import SerialSparseExecutor
from ggce.executors.parallel import ParallelDenseExecutor
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS

from ggce import Model, System, DenseSolver  # noqa

# read key parameters from command line call

# define the k/w grids
k = 0.5 * np.pi
eta = 0.05
dw = eta / 10.
wgrid = np.arange(-2, 5+dw, dw)
lam = 1.

# initialize the GGCE and Hamiltonian model parameters
M, N, M2, N2 = 2, 5, 2, 5

# Check the T = 0 case
model_zero = Model.from_parameters(hopping = 1.0, temperature = 0.0)
model.add_("Holstein", 1., M, N, M2, N2, dimensionless_coupling_strength = lam)
model.add_("Peierls", 2., M, N, M2, N2, dimensionless_coupling_strength = lam)

solver =


a, b, c = float(sys.argv[10]), float(sys.argv[11]), float(sys.argv[12])
coupling_H, Omega_H = a, b
temperature = c * Omega_H

# these remain static at least in these calculations
hopping = 1.
model_type = "Holstein"

COMM = MPI.COMM_WORLD
if COMM.Get_rank() == 0:
    cpus_per_task = int(os.environ["OMP_NUM_THREADS"])
    worldsize = COMM.Get_size()
    cpus_total = cpus_per_task * worldsize
    if server_name == "lisa": # on lisa have 28 cores per node
        nodes = int(cpus_total / 28)
    tasks = worldsize
    nrc_label = f"{nodes}/{tasks}/{cpus_per_task}"

# create (or overwrite) output directory
results_dir = os.path.join(script_dir, f'spect_func_hol_finT_{M}{N}_{M2}{N2}')
if COMM.Get_rank() == 0:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        os.mkdir(results_dir)

# initialize the model
model = Model(calc_type, "info", log_file=None)
model.set_parameters(hopping=hopping, temperature = temperature, \
            lattice_constant=1.0, dimension=1, max_bosons_per_site=None)
model.add_coupling(
    model_type, Omega=Omega_H, M=cloud_ext_H, N=bosons_H, \
    M_tfd = cloud_ext_tfd_H, N_tfd = bosons_tfd_H,
    dimensionless_coupling=coupling_H)
model.finalize()

# create executor to run the calculations
# executor = ParallelSparseExecutorMUMPS(model, "debug", mpi_comm=COMM)
executor = ParallelDenseExecutor(model, "info", mpi_comm=COMM)
executor.prime()

executor._logger.info(f"Running a calculation of type {calc_type}.")
executor._logger.info(f"kgrid is {kgrid}.")
executor._logger.info(f"wgrid is {wgrid}.")
executor._logger.info(f"eta is {eta}.")
executor._logger.info(f"Model type is {model_type}.")
executor._logger.info(f"Hopping t is {hopping}.")
executor._logger.info(f"El-phon coupling t is {coupling_H}.")
executor._logger.info(f"Omega_H is {Omega_H}.")
executor._logger.info(f"Temperature is {c} in units of Omega_H.")
executor._logger.info(f"Cloud is defined by (M, N, M_t, N_t) = ({M},{N},{M2},{N2}).")

start = time.time()
all_res = executor.spectrum(kgrid, wgrid, eta, return_meta = True, return_G = False)
dt = time.time() - start

if COMM.Get_rank() == 0:

    # print model parameters to screen for posterity
    model.visualize()
    av_time_per_point = dt / (len(kgrid) * len(wgrid))
    matr_size = sum([len(val) for val in executor._system.equations.values()])

    res, meta = all_res
    xx = np.array( res ).T
    # create the output file
    # if this is the first time then create the file with the heading
    result_file = os.path.join(results_dir, f"results_finite_t.out")
    np.savetxt(result_file, xx,\
                header = f"momentum horizontal, frequency vertical, total time {dt:.2f} s",\
                delimiter = '    ')

    result_file_wgrid = os.path.join(results_dir, f"wgrid.out")
    np.savetxt(result_file_wgrid, wgrid,\
                header = f"frequency",\
                delimiter = '    ')

    result_file_kgrid = os.path.join(results_dir, f"kgrid.out")
    np.savetxt(result_file_kgrid, kgrid,\
                header = f"momentum",\
                delimiter = '    ')

    result_reporting = os.path.join(results_dir, f"reporting.csv")
    datetoday = str(date.today())
    key_reporting = np.array([[datetoday, calc_type, server_name, jobID, nrc_label, \
                        eta, coupling_H, Omega_H, temperature / Omega_H, \
                        M, N, M2, N2, matr_size, av_time_per_point],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    np.savetxt(result_reporting, key_reporting, delimiter = ',',\
                header = f"Date,Type,Server,ID,N/R/C,Eta,Lam,O_H"\
                    f",T,M,N,M_t,N_t,Size,Time", \
                    fmt = '%s')
