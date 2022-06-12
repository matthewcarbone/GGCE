import numpy as np

from ggce import logger
from ggce.utils.utils import peak_location_and_weight, \
                        peak_location_and_weight_wstep, chunk_jobs, \
                                                            float_to_list


class ExecutorResults:

    def __init__(self):
        ...


class BaseExecutor:
    """Base executor class.

    Parameters
    ----------
    model: ggce.model.Model
    default_console_logging_level: str
        The default level for initializing the logger. (The default is
        'INFO', which will log everything at info or above).
    log_file: str
        Location of the log file. (The default is None, which defaults to
        no log file being produced).
    mpi_comm: mpi4py.MPI.Intracomm, optional
        The MPI communicator accessed via MPI.COMM_WORLD. (The default is
        None, which is taken to imply a single MPI process).
    log_every : int, optional
        Determines how often to log calculation results at the info level (the
        default is 1).
    """

    @property
    def system(self):
        return self._system

    @property
    def mpi_comm(self):
        return self._mpi_comm

    @property
    def mpi_rank(self):
        if self._mpi_comm is not None:
            return self._mpi_comm.Get_rank()
        return 0

    @property
    def mpi_world_size(self):
        if self._mpi_comm is not None:
            return self._mpi_comm.Get_size()
        return 1

    def __init__(self, system, mpi_comm=None):
        # Initialize the executor's logger and adjust the default logging level
        # for the console output

        self._mpi_comm = mpi_comm
        self._system = system
        self._basis = None
        self._total_jobs_on_this_rank = 1

    def get_jobs_on_this_rank(self, jobs):
        """Gets the jobs assigned to this rank. Note this method silently
        behaves as it should when the world size is 1 (or there's no MPI
        communicator).

        Parameters
        ----------
        jobs : list
            The jobs to chunk.

        Returns
        -------
        list
            The jobs assigned to this rank.
        """

        if self.mpi_comm is None:
            return jobs

        return chunk_jobs(jobs, self.mpi_world_size, self.mpi_rank)

    def _log_job_distribution_information(self, jobs_on_rank):
        if self.mpi_comm is None:
            return
        all_jobs = self.mpi_comm.gather(jobs_on_rank, root=0)
        if self.mpi_rank == 0:
            lengths = np.array([len(xx) for xx in all_jobs])  # Get job lengths
            std = np.std(lengths)
            mu = np.mean(lengths)
            if std == 0:
                mu = int(mu)
                logger.info(f"Jobs balanced on all ranks ({mu}/rank)")
            else:
                logger.info(
                    f"Job balance: {mu:.02f} +/- {std:.02f} per rank"
                )
        self.mpi_comm.Barrier()

    def _log_spectral_error(self, k, w):
        logger.error(
            f"Negative spectral weight at k, w = ({k:.02f}, {w:.02f}"
        )

    def _log_current_status(self, k, w, A, index, dt):

        # if running in serial, this prevents an error thrown on trying to
        # pass NoneType a formatting string
        if self._total_jobs_on_this_rank is None:
            return None

        if index is None:
            index = -1
        index += 1
        msg1 = f"({index:03}/{self._total_jobs_on_this_rank:03}) solved "
        msg_debug = f"{msg1} A({k:.02e}, {w:.02e}) = {A:.02e}"
        if index % self._log_every == 0:
            logger.info(msg_debug, elapsed=dt)
        logger.debug(msg_debug, elapsed=dt)

    def _scaffold():
        raise NotImplementedError

    def solve():
        raise NotImplementedError

    def spectrum(self, k, w, eta, **solve_kwargs):
        """Solves for the spectrum in serial.

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
        np.ndarray, dict
            The resultant spectrum, or resultant spectrum and meta information.
        """

        k = float_to_list(k)
        w = float_to_list(w)

        self._total_jobs_on_this_rank = len(k) * len(w)

        # This orders the jobs such that when constructed into an array, the
        # k-points are the rows and the w-points are the columns after reshape
        jobs = [(_k, _w) for _k in k for _w in w]

        s = [
            self.solve(_k, _w, eta, ii, **solve_kwargs)
            for ii, (_k, _w) in enumerate(jobs)
        ]

        # Separate meta information
        res = [xx[0] for xx in s]
        meta = [xx[1] for xx in s]

        if return_G:
            res = np.array(res)
        else:
            res = -np.array(res).imag / np.pi
        res = res.reshape(len(k), len(w))

        if return_meta:
            return (res, meta)
        return res

    def dispersion(
        self, kgrid, w0, eta, eta_div=3.0, eta_step_div=5.0,
        next_k_offset_factor=1.5, nmax=1000, peak_routine = "change_eta", **solve_kwargs
    ):
        """Computes the dispersion of the peak closest to the provided w0 by
        assuming that the peak is Lorentzian in nature. This allows us to
        take two points, each at a different value of the broadening, eta, and
        compute the location of the Lorentzian (ground state energy) and
        quasi-particle weight exactly, at least in principle. As stated, we
        rely on the assumption that the peak is Lorentzian, which is only true
        in some cases (e.g. the polaron).

        This method works as follows: (1) An initial guess for the peak
        location of the first entry in kgrid is provided (w0). (2) The location
        of the peak is found by slowly increasing w in increments of
        eta / eta_step_div until the first time the value of A decreases from
        the previou sone. (3) The location is found (as is the weight) by
        computing A using a second broadening given by eta / eta_div. (4) This
        value is logged in results, and the algorithm moves to the next
        k-point. The new initial guess for the next peak location is given by
        the found location of the previous k-point minus
        eta * next_k_offset_factor.

        UPDATE: The method can now be run using PETSc "ParallelSparse" protocol.
        It is parallel in that the for a single (k,w) point, the matrix is
        distributed across different tasks: however, it is "serial" in that
        it still works its way through one (k,w) point at a time. If you try to
        call this using ParallelDenseExecutor you will get a NotImplementedError.

        Parameters
        ----------
        kgrid : list
            A list of the k-points to calculate.
        w0 : float
            The initial guess for the peak location for the first k-point only.
        eta : float
            The broadening parameter.
        eta_div : float, optional
            Used for the computation of the second A value (the default is
            3.0, a good empirical value).
        eta_step_div : float, optional
            Defines the step in frequency space as eta / eta_step_div (the
            default is 5.0).
        next_k_offset_factor : float, optional
            Defines how far back from the found peak location to start the
            algorithm at the next k-point. The next start location is given by
            the found location minus eta * next_k_offset_vactor (the default is
            1.5).
        nmax : int, optional
            The maximum number of steps to take in eta before gracefully
            erroring out and returning the previously found values (the
            default is 1000).

        Returns
        -------
        list
            List of dictionaries, each of which contains 5 keys: the k-value at
            which the calculation was run ('k'), lists for the w-values and
            spectrum values ('w' and 'A'), and the ground state energy and
            quasi-particle weight ('ground_state' and 'weight').
        """

        results = []
        w_val = w0
        nk = len(kgrid)
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
                    logger.error("Exceeded maximum omega points")
                    return results

                G, _ = self.solve(k_val, w_val, eta, **solve_kwargs)
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

                # This is a maximum, run the calculation again one dw step prior to this
                if peak_routine == "change_eta":
                    eta_prime = eta / eta_step_div
                    G2, _ = self.solve(k_val, w_val, eta_prime, **solve_kwargs)
                    A2 = -G2.imag / np.pi
                    loc, weight = peak_location_and_weight(w_val, A, A2, eta, eta_prime)
                elif peak_routine == "change_w":
                    w_val_prime = w_val - 2.*eta / eta_step_div
                    G2, _ = self.solve(k_val, w_val_prime, eta, **solve_kwargs)
                    A2 = -G2.imag / np.pi
                    loc, weight = peak_location_and_weight_wstep(w_val, \
                                                    w_val_prime, A, A2, eta)
                results[ii]['ground_state'] = loc
                results[ii]['weight'] = weight
                w_val = loc - eta * next_k_offset_factor
                logger.info(
                    f"For k ({ii:03}/{nk:03}) = {k_val:.02f}: GS={loc:.08f}, "
                    f"wt={weight:.02e}"
                )
                break

        return results
