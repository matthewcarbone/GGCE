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
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS
from ggce.executors.petsc4py.doubleparallel import DoubleParallelExecutorMUMPS
from ggce.utils.utils import setup_directory
from ggce.utils.utils import get_grids_from_files

''' CREATE COMMUNICATOR AND SETUP INPUT / OUTPUT DIRECTORY '''

COMM = MPI.COMM_WORLD

M, N, M2, N2 = 3, 8, 2, 12
basis_dir = os.path.join(script_dir, f'basis_gen_{M}{N}_{M2}{N2}')
solution_dir = os.path.join(script_dir, f'HfT_{M}{N}_{M2}{N2}')
if COMM.Get_rank() == 0:
    setup_directory(solution_dir)

''' GET THE WGRID AND KGRID FROM THE FILES IN THE DIRECTORY'''
kgrid, wgrid, eta = get_grids_from_files(basis_dir)

''' LOAD MODEL AND INITIALIZE EXECUTOR '''
model_loc = os.path.join(basis_dir, "model.mdl")
with open(model_loc, "rb") as modelfile:
    model = pickle.load(modelfile)

executor = DoubleParallelExecutorMUMPS(model, "debug", brigade_size=6, mpi_comm=COMM)
executor.set_input_dir(basis_dir)
executor.prime_from_disk()

start = time.time()
all_res = executor.spectrum(kgrid, wgrid, eta, \
                                    return_meta = True, return_G = False)
dt = time.time() - start
if COMM.Get_rank() == 0:
    executor._logger.info(f"Took {dt:.2f} seconds.")
    model.visualize()

    res, meta = all_res
    xx = np.array( res ).T
    # create the output file
    # if this is the first time then create the file with the heading
    result_file = os.path.join(solution_dir, f"results.out")
    np.savetxt(result_file, xx,\
                header = f"momentum horizontal, frequency vertical, total time {dt:.2f} s",\
                delimiter = '    ')

    result_file_wgrid = os.path.join(solution_dir, f"wgrid.out")
    np.savetxt(result_file_wgrid, wgrid,\
                header = f"frequency",\
                delimiter = '    ')

    result_file_kgrid = os.path.join(solution_dir, f"kgrid.out")
    np.savetxt(result_file_kgrid, kgrid,\
                header = f"momentum",\
                delimiter = '    ')
