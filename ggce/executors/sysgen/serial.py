#!/usr/bin/env python3

import numpy as np
import time
import os

from ggce.engine.system import System
from ggce.utils.logger import Logger
from ggce.utils.utils import peak_location_and_weight, chunk_jobs, \
    float_to_list
from ggce.executors.serial import SerialSparseExecutor


class SerialSystemGenerator(SerialSparseExecutor):
    """System generator class to make systems of equations and
    dump them to disk for processing with PETSc."""

    def __init__(
        self, model, default_console_logging_level='INFO',
        log_file=None, mpi_comm=None, log_every=1, basis_dir=None):
        # Initialize the executor's logger and adjust the default logging level
        # for the console output
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_world_size = 1
        if mpi_comm is not None:
            self.mpi_comm = mpi_comm
            self.mpi_rank = mpi_comm.Get_rank()
            self.mpi_world_size = mpi_comm.Get_size()
        self._logger = Logger(log_file, mpi_rank=self.mpi_rank)
        self._logger.adjust_logging_level(default_console_logging_level)
        self._model = model
        self._system = None
        self._basis = None
        self._log_every = log_every
        self._total_jobs_on_this_rank = 1

        self.basis_dir = basis_dir

    def solve(self, k, w, eta, index=None,**kwargs):

        return NotImplementedError

    def prepare(self, k, w, eta, index=None,**kwargs):
        """Prepare the sparse-represented system to be solved by another
        executor.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.
        index : int, optional
            The calculation index (the default is None).

        Returns
        -------
        np.array, dict
            A fake value of G (-1, so that it is obvious this is an error),
            the time elapsed to prepare the system for this (k, w) point.
        """

        t0 = time.time()
        row_ind, col_ind, dat = self._sparse_matrix_from_equations(k, w, eta)
        dt = time.time() - t0
        xx = np.array([row_ind, col_ind, dat]).T

        basis_loc = os.path.join(self.basis_dir, f"k_{k:.2f}_w_{w:.3f}_e_{eta:.2f}.bss")
        with open(basis_loc, "wb") as basis_file:
            np.savetxt(basis_loc, xx, delimiter = ",", \
                        header = f"Matrix for k = {k}, w = {w}, eta = {eta} "\
                        f"in CSR format: row, column, datum", \
                        fmt = '%s')

        return np.array(-1), {"time": [dt]}
