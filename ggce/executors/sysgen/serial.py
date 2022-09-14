#!/usr/bin/env python3

import numpy as np
import time
import os
import pickle

from ggce.engine.system import System
from ggce.utils.logger import Logger
from ggce.utils.utils import peak_location_and_weight, chunk_jobs, \
    float_to_list
from ggce.executors.serial import SerialSparseExecutor


class SerialSystemGenerator(SerialSparseExecutor):
    """System generator class to make systems of equations and
    dump them to disk for processing with PETSc."""

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

        return np.array(-1), {"time": [dt]}

    def prepare_spectrum(self, k, w, eta, **solve_kwargs):
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

        self._total_jobs_on_this_rank = len(k) * len(w)

        # This orders the jobs such that when constructed into an array, the
        # k-points are the rows and the w-points are the columns after reshape
        jobs = [(_k, _w) for _k in k for _w in w]

        s = [
            self.prepare(_k, _w, eta, **solve_kwargs)
            for ii, (_k, _w) in enumerate(jobs)
        ]

        # Separate meta information
        res = [xx[0] for xx in s]
        meta = [xx[1] for xx in s]

        return (res, meta)
