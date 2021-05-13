#!/usr/bin/env python3

from itertools import islice
import numpy as np

from ggce.engine.system import System
from ggce.engine.structures import ParameterObject
from ggce.utils.logger import Logger


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

    def spectrum(self, k, w, eta=None):
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
