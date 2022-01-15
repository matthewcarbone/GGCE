import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
import time
import pickle

script_dir = f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"\
            f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expansion/"\
            f"sysgen/scripts"
ggce_head_dir = f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"\
            f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expansion/"\
            f"sysgen"

sys.path.append(ggce_head_dir)
import ggce

from mpi4py import MPI

from ggce.model import Model  # noqa: E402
from ggce.executors.sysgen.parallel import ParallelSystemGenerator
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS
from ggce.utils.utils import setup_directory

''' DEFINING MODEL PARAMETERS '''
# M, N, M2, N2 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

M, N, M2, N2 = 2,8,2,8
# a, b, c = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
a, b, c = 1., 0.5, 0.8

cloud_ext_H, cloud_ext_tfd_H = M, M2
bosons_H, bosons_tfd_H = N, N2

coupling_H = a
Omega_H = b
temperature = c * Omega_H

hopping = 1.
eta = 0.05

model_type1 = "Holstein"

# define the k/w grids for the dispersion calculation
kgrid = np.linspace(0, np.pi/2., 4)
w_left, w_right = -4., -3.970
dw = eta / 10.
wgrid = np.arange(w_left, w_right+dw, dw)

try:
    assert (1-len(wgrid)*len(kgrid) % 2)
except AssertionError:
    print(f"Number of jobs ({len(wgrid)*len(kgrid)}) will be non-even. "
        f"If you intend to use MPI brigades, change it to be even. Exiting.")
    exit()

''' CREATE COMMUNICATOR AND SETUP OUTPUT DIRECTORY '''

COMM = MPI.COMM_WORLD

basis_dir = os.path.join(script_dir, f'basis_gen_{M}{N}_{M2}{N2}')
if COMM.Get_rank() == 0:
    setup_directory(basis_dir)

''' INITIALIZE MODEL AND EXECUTOR '''

model = Model("model", "info", log_file=None)
model.set_parameters(hopping=hopping, temperature = temperature, \
            lattice_constant=1.0, dimension=1, max_bosons_per_site=None)
model.add_coupling(
    model_type1, Omega=Omega_H, M=cloud_ext_H, N=bosons_H, \
    M_tfd = cloud_ext_tfd_H, N_tfd = bosons_tfd_H,
    dimensionless_coupling=coupling_H)
model.finalize()

sysgen = ParallelSystemGenerator(model, "debug", mpi_comm = COMM)
sysgen.set_output_dir(basis_dir)
sysgen.prime()

start = time.time()
all_res = sysgen.prepare_spectrum(kgrid, wgrid, eta)
dt = time.time() - start

if COMM.Get_rank() == 0:
    sysgen._logger.info(f"All systems prepared. This took {dt:.2f} sec.")
    model_loc = os.path.join(basis_dir, "model.mdl")
    with open(model_loc, "wb") as modelfile:
        pickle.dump(model, modelfile)
