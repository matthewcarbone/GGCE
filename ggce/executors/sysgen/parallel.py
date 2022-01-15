#!/usr/bin/env python3

import numpy as np
import time
import os
import pickle

from ggce.engine.system import System
from ggce.utils.logger import Logger
from ggce.utils.utils import peak_location_and_weight, chunk_jobs, \
    float_to_list
from ggce.executors.sysgen.serial import SerialSystemGenerator


class ParallelSystemGenerator(SerialSystemGenerator):
    """System generator class to make systems of equations and
    dump them to disk for running the calculation later.

    This helps separate the memory-expensive basis generation
    from the matrix solution and avoid clogging up the memory
    with duplicates of the basis that each rank holds.

    This particular class allows to do the preparation in parallel."""

    def set_output_dir(self, basis_dir):

        self.basis_dir = basis_dir

    def solve(self, k, w, eta, index=None,**kwargs):

        return NotImplementedError

    def prepare(self, k, w, eta, **kwargs):
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

        Returns
        -------
        np.array, dict
            A fake value of G (-1, so that it is obvious this is an error),
            the time elapsed to prepare the system for this (k, w) point.
        """

        t0 = time.time()
        row_ind, col_ind, dat = self._sparse_matrix_from_equations(k, w, eta)
        dt = time.time() - t0
        xx = [row_ind, col_ind, dat]

        basis_loc = os.path.join(self.basis_dir, \
                                        f"k_{k}_w_{w}_e_{eta}.bss")
        with open(basis_loc, "wb") as basis_file:
            pickle.dump(xx, basis_file)
        ## this return statement is needed to avoid raising errors,
        ## but made -1 to be clearly incorrect
        return np.array(-1), {"time": [dt]}

    def prepare_spectrum(self, k, w, eta, return_meta=False, **solve_kwargs):
        """Prepares matrices for the spectrum in serial.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.
        **solve_kwargs
            Extra arguments to pass to solve().

        Returns
        -------
        np.array, dict
            A fake value of G (-1, so that it is obvious this is an error),
            the time elapsed to prepare the system for this (k, w) point.
        """

        k = float_to_list(k)
        w = float_to_list(w)

        # This orders the jobs such that when constructed into an array, the
        # k-points are the rows and the w-points are the columns after reshape
        jobs = [(_k, _w) for _k in k for _w in w]

        jobs_on_rank = self.get_jobs_on_this_rank(jobs)
        self._logger.debug(f"{len(jobs_on_rank)} jobs todo")
        self._log_job_distribution_information(jobs_on_rank)
        self._total_jobs_on_this_rank = len(jobs_on_rank)

        s = [
            self.prepare(_k, _w, eta, **solve_kwargs)
            for ii, (_k, _w) in enumerate(jobs_on_rank)
        ]

        # Gather the results on rank 0 -- not that it matters since
        # this is only basis preparation
        all_results = self.mpi_comm.gather(s, root=0)

        if self.mpi_rank == 0:

            all_results = [xx[ii] for xx in all_results for ii in range(len(xx))]
            s = [xx[0] for xx in all_results]
            meta = [xx[1] for xx in all_results]
            res = np.array(s)

            # Ensure the returned array has the proper shape
            res = res.reshape(len(k), len(w))
            if return_meta:
                return (res, meta)
            return res
