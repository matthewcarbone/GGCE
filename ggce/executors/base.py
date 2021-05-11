#!/usr/bin/env python3

from itertools import islice
import numpy as np

from ggce.engine.system import System
from ggce.engine.structures import ParameterObject
from ggce.utils.logger import Logger
from ggce.utils.utils import peak_location_and_weight


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
    mpi_comm: mpi4py.MPI.Intracomm, optional
        The MPI communicator accessed via MPI.COMM_WORLD. (The default is
        None, which is taken to imply a single MPI process).
    """

    def __init__(
        self, parameter_dict, default_console_logging_level='INFO',
        log_file=None, mpi_comm=None,
    ):
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
        self._parameter_dict = parameter_dict
        self._parameters = None
        self._system = None
        self._basis = None

    def get_jobs_on_this_rank(self, jobs):
        """Get's the jobs assigned to this rank. Note this method silently
        behaves as it should when the world size is 1, and will log a warning
        if it is called but the communicator is not initialized.

        Parameters
        ----------
        jobs : list
            The jobs to chunk

        Returns
        -------
        list
            The jobs assigned to this rank.
        """

        if self.mpi_comm is None:
            self._logger.warning("Chunking jobs with COMM_WORLD_SIZE=1")
            return jobs

        it = iter(jobs)
        return list(iter(lambda: tuple(islice(it, self.size)), ()))[self.rank]

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

    def _log_spectral_error(self, k, w):
        self._logger.error(
            f"Negative spectral weight at k, w = ({k:.02f}, {w:.02f}"
        )

    def _scaffold():
        raise NotImplementedError

    def prime():
        raise NotImplementedError

    def solve():
        raise NotImplementedError

    def compute(self, k, w, eta=None):
        """Solves for the spectrum in serial.

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
        np.ndarray
            The resultant spectrum.
        """

        if isinstance(k, float):
            k = [k]
        if isinstance(w, float):
            w = [w]

        s = [[-self.solve(_k, _w)[0].imag / np.pi for _w in w] for _k in k]
        return np.array(s)

    def band(
        self, kgrid, w0, eta_div=3.0, eta_step_div=5.0,
        next_k_offset_factor=1.5, eta=None, nmax=1000
    ):
        """[summary]

        [description]

        Parameters
        ----------
        kgrid : {[type]}
            [description]
        w0 : {[type]}
            [description]
        eta_div : {[type]}
            [description]
        eta_step_div : {number}, optional
            [description] (the default is 3.0, which [default_description])
        next_k_offset_factor : {number}, optional
            [description] (the default is 1.5, which [default_description])
        eta : {[type]}, optional
            [description] (the default is None, which [default_description])
        nmax : {number}, optional
            [description] (the default is 1000, which [default_description])

        Returns
        -------
        [type]
            [description]
        """

        finfo = self._parameters.get_fFunctionInfo()
        eta = eta if eta is not None else finfo.eta

        if self.mpi_comm is not None:
            self._logger.critical(
                "Band calculations should be run using a serial protocol"
            )
            self.mpi_comm.Abort()

        results = []
        w_val = w0
        for ii, k_val in enumerate(kgrid):

            current_n_w = 0
            reference = 0.0

            results.append({
                'k': k_val,
                'w': [],
                'A': [],
                'ground_state': None,
                'weight': None
            })

            while True:

                if current_n_w > nmax:
                    self._logger.error("Exceeded maximum omega points")
                    return results

                G, _ = self.solve(k_val, w_val, eta=eta)
                A = -G.imag / np.pi
                results[ii]['w'].append(w_val)
                results[ii]['A'].append(A)

                # Check and see whether or not we've found a local maxima
                if reference < A:

                    # This is not a maximum
                    reference = A

                    current_n_w += 1
                    w_val += eta / eta_step_div
                    continue

                # This is a maximum, run the calculation again using eta prime
                eta_prime = eta / eta_div
                G2, _ = self.solve(k_val, w_val, eta=eta_prime)
                A2 = -G2.imag / np.pi
                loc, weight = peak_location_and_weight(
                    w_val, A, A2, eta, eta_prime
                )
                results[ii]['ground_state'] = loc
                results[ii]['weight'] = weight
                w_val = loc - eta * next_k_offset_factor
                break

        return results
