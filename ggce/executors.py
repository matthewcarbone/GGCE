#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

from ggce.engine.system import System
from ggce.engine.structures import ParameterObject
from ggce.engine.physics import G0_k_omega
from ggce.utils.logger import Logger
from ggce.utils.utils import elapsed_time_str


BYTES_TO_MB = 1048576


class BaseExecutor:
    """Base executor class.

    Parameters
    ----------
    parameter_dict: dict
        Dictionary of the parameters used to initialize the
        ParameterObject.
    default_console_logging_level: str
        The default level for initializing the logger. (The default is
        'INFO', which will log everything at info or above).
    log_file: str
        Location of the log file. (The default is None, which defaults to
        no log file being produced).
    """

    def __init__(
        self, parameter_dict, default_console_logging_level='INFO',
        log_file=None
    ):
        # Initialize the executor's logger and adjust the default logging level
        # for the console output
        self._logger = Logger(log_file)
        self._logger.adjust_logging_level(default_console_logging_level)
        self._parameter_dict = parameter_dict
        self._parameters = None
        self._system = None
        self._basis = None

    def get_parameters(self, return_dict=False):
        """Returns the parameter information.

        Parameters
        ----------
        return_dict: bool
            If True, returns the dictionary used to initialize the
            ParameterObject, else returns the ParameterObject instance itself,
            which will be None if _prime_parameters() was not called. (The
            default is False).

        Returns
        -------
        dict or ParameterObject or None
        """

        if return_dict:
            return self._parameter_dict
        return self._parameters

    def get_system(self):
        """Returns the system object, which will be None if _prime_system()
        has not been called.

        Returns
        -------
        System
        """

        return self._system

    def _prime_parameters(self):

        self._parameters = ParameterObject(self._parameter_dict, self._logger)
        self._parameters.prime()

    def _prime_system(self):

        self._system = System(self._parameters, self._logger)
        self._system.initialize_generalized_equations()
        self._system.initialize_equations()
        self._system.generate_unique_terms()

    def prime():
        raise NotImplementedError

    def scaffold():
        raise NotImplementedError

    def solve():
        raise NotImplementedError


class SerialSparseExecutor(BaseExecutor):
    """Uses the SciPy sparse solver engine to solve for G(k, w) in serial."""

    def prime(self):
        self._prime_parameters()
        self._prime_system()
        self._basis = self._system.get_basis(full_basis=True)

    def scaffold(self, k, w, eta=None):
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

        _dt, _fmt = elapsed_time_str(dt)
        self._logger.debug(f"Scaffold complete in {_dt:.01f} {_fmt}")

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

        X, v = self.scaffold(k, w, eta)

        # Bottleneck: solve the matrix
        t0 = time.time()
        res = spsolve(X, v)
        dt = time.time() - t0
        _dt, _fmt = elapsed_time_str(dt)
        self._logger.debug(f"Sparse matrices solved in {_dt:.01f} {_fmt}")

        G = res[self._basis['{G}(0.0)']]

        if -G.imag / np.pi < 0.0:
            self._logger.error(
                f"Negative A({k:.02f}, {w:.02f}): {(-G.imag / np.pi):.02f}"
            )

        return np.array(G), {'time': [dt]}


class SerialDenseExecutor(BaseExecutor):
    """Uses the SciPy dense solver engine to solve for G(k, w) in serial. This
    method uses the continued fraction approach,

    .. math:: R_{n-1} = (1 - \\beta_{n-1}R_{n})^{-1} \\alpha_{n-1}

    with

    .. math:: R_n = \\alpha_n

    and

    .. math:: f_n = R_n f_{n-1}
    """

    def prime(self):
        self._prime_parameters()
        self._prime_system()
        self._basis = self._system.get_basis(full_basis=False)

    def _fill_matrix(self, k, w, n_phonons, shift, eta):

        n_phonons_shift = n_phonons + shift

        equations_n = self._system.equations[n_phonons]

        # Initialize a matrix to fill
        d1 = len(self._basis[n_phonons])
        d2 = len(self._basis[n_phonons + shift])
        A = np.zeros((d1, d2), dtype=np.complex64)

        # Fill the matrix of coefficients
        for ii, eq in enumerate(equations_n):
            index_term_id = eq.index_term.identifier()
            ii_basis = self._basis[n_phonons][index_term_id]
            for term in eq.terms_list:
                if term.get_total_bosons() != n_phonons_shift:
                    continue
                t_id = term.identifier()
                jj_basis = self._basis[n_phonons_shift][t_id]
                A[ii_basis, jj_basis] += term.coefficient(k, w, eta)

        return A

    def _get_alpha(self, k, w, n_phonons, eta=None):

        t0 = time.time()
        A = self._fill_matrix(k, w, n_phonons, -1, eta)
        dt = time.time() - t0
        _dt, _fmt = elapsed_time_str(dt)
        self._logger.debug(f"Filled alpha in {_dt:.01f} {_fmt}")
        return A

    def _get_beta(self, k, w, n_phonons, eta=None):

        t0 = time.time()
        A = self._fill_matrix(k, w, n_phonons, 1, eta)
        dt = time.time() - t0
        _dt, _fmt = elapsed_time_str(dt)
        self._logger.debug(f"Filled beta in {_dt:.01f} {_fmt}")
        return A

    def solve(self, k, w, eta=None):
        """Solve the dense-represented system.

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

        t0_all = time.time()

        meta = {
            'alphas': [],
            'betas': [],
            'inv': [],
            'time': []
        }

        total_phonons = np.sum(self._parameters.N)

        for n_phonons in range(total_phonons, 0, -1):

            # Special case of the recursion where R_N = alpha_N.
            if n_phonons == total_phonons:
                R = self._get_alpha(k, w, n_phonons, eta=eta)
                meta["alphas"].append(R.shape)
                continue

            # Get the next loop's alpha and beta values
            beta = self._get_beta(k, w, n_phonons, eta=eta)
            meta["betas"].append(beta.shape)
            alpha = self._get_alpha(k, w, n_phonons, eta=eta)
            meta["alphas"].append(alpha.shape)

            # Compute the next R
            X = np.eye(beta.shape[0], R.shape[1]) - beta @ R
            meta["inv"].append(X.shape[0])
            t0 = time.time()
            R = linalg.solve(X, alpha)
            dt = time.time() - t0
            _dt, _fmt = elapsed_time_str(dt)
            self._logger.debug(
                f"Inverted [{X.shape}, {alpha.shape}] in {_dt:.01f} {_fmt}"
            )
            meta["time"].append(dt)

        finfo = self._parameters.get_fFunctionInfo()
        G0 = G0_k_omega(k, w, finfo.a, finfo.eta, finfo.t)

        beta0 = self._get_beta(k, w, 0, eta=eta)
        result = (G0 / (1.0 - beta0 @ R)).squeeze()

        dt = time.time() - t0_all
        _dt, _fmt = elapsed_time_str(dt)

        self._logger.debug(f"Solve complete in {_dt:.01f} {_fmt}")

        return np.array(result, dtype=np.complex64), meta
