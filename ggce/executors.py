#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import OrderedDict
import copy
import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

from ggce.engine.system import System
from ggce.engine.structures import ParameterObject
from ggce.utils.logger import Logger
from ggce.utils.utils import elapsed_time_str


BYTES_TO_MB = 1048576


class BaseExecutor:

    def __init__(
        self, parameter_dict, default_console_logging_level='INFO',
        log_file=None
    ):
        """Initializes the BaseExecutor.

        Parameters
        ----------
        parameter_dict : {dict}
            Dictionary of the parameters used to initialize the
            ParameterObject.
        default_console_logging_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR'},
        optional
            The default level for initializing the logger. (The default is
            'INFO', which will log everything at info or above).
        log_file : {str}, optional
            Location of the log file. (The default is None, which defaults to
            no log file being produced).
        """

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
        return_dict : {bool}, optional
            If True, returns the dictionary used to initialize the
            ParameterObject, else returns the ParameterObject instance itself,
            which will be None if _prime_parameters() was not called. (The
            default is False).

        Returns
        -------
        dict, ggce.engine.structures.ParameterObject
        """

        if return_dict:
            return self._parameter_dict
        return self._parameters

    def get_system(self):
        """Returns the system object, which will be None if _prime_system()
        has not been called.

        Returns
        -------
        ggce.engine.system.System
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
        for n_bosons in range(total_bosons + 1):
            for eq in self._system.equations[n_bosons]:
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
        float, dict
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

        return G, {'time': [dt]}
