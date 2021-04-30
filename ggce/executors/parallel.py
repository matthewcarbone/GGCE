#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

from ggce.executors.base import BaseExecutor
from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega


BYTES_TO_MB = 1048576


class ParallelSparseExecutor(BaseExecutor, SerialSparseExecutor):
    """Uses the SciPy sparse solver engine to solve for G(k, w) in parallel."""

    def prime(self):
        self._prime_parameters()
        self._prime_system()
        self._basis = self._system.get_basis(full_basis=True)

    def _scaffold(self, k, w, eta=None):
        """Prepare the X, v sparse representation of the matrix to solve.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation (the default
            is None, which uses the value provided in parameter_dict at
            instantiation).

        Returns
        -------
        csr_matrix, csr_matrix
            Sparse representation of the matrix equation to solve, X and v.
        """

        t0 = time.time()

        row_ind = []
        col_ind = []
        dat = []

        total_bosons = np.sum(self._parameters.N)
        for n_phonons in range(total_bosons + 1):
            for eq in self._system.equations[n_phonons]:
                row_dict = dict()
                index_term_id = eq.index_term.identifier()
                ii_basis = self._basis[index_term_id]

                for term in eq.terms_list + [eq.index_term]:
                    jj = self._basis[term.identifier()]
                    try:
                        row_dict[jj] += term.coefficient(k, w, eta)
                    except KeyError:
                        row_dict[jj] = term.coefficient(k, w, eta)

                row_ind.extend([ii_basis for _ in range(len(row_dict))])
                col_ind.extend([key for key, _ in row_dict.items()])
                dat.extend([value for _, value in row_dict.items()])

        X = coo_matrix((
            np.array(dat, dtype=np.complex64),
            (np.array(row_ind), np.array(col_ind))
        )).tocsr()

        size = (X.data.size + X.indptr.size + X.indices.size) / BYTES_TO_MB

        self._logger.debug(f"Memory usage of sparse X is {size:.01f} MB")

        # Initialize the corresponding sparse vector
        # {G}(0)
        row_ind = np.array([self._basis['{G}(0.0)']])
        col_ind = np.array([0])
        v = coo_matrix((
            np.array(
                [self._system.equations[0][0].bias(k, w, eta)],
                dtype=np.complex64
            ), (row_ind, col_ind)
        )).tocsr()

        dt = time.time() - t0

        self._logger.debug("Scaffold complete", elapsed=dt)

        return X, v

    def solve(self, k, w, eta=None):
        """Solve the sparse-represented system.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation (the default
            is None, which uses the value provided in parameter_dict at
            instantiation).

        Returns
        -------
        np.ndarray, dict
            The value of G and meta information, which in this case, is only
            the time elapsed to solve for this (k, w) point.
        """

        X, v = self._scaffold(k, w, eta)

        # Bottleneck: solve the matrix
        t0 = time.time()
        res = spsolve(X, v)
        dt = time.time() - t0
        self._logger.debug("Sparse matrices solved", elapsed=dt)

        G = res[self._basis['{G}(0.0)']]

        if -G.imag < 0.0:
            self._log_spectral_error(k, w)

        return np.array(G), {'time': [dt]}
