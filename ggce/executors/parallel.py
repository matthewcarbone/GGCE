import numpy as np

from ggce.executors.serial import SerialDenseExecutor
from ggce.utils.utils import float_to_list


class ParallelDenseExecutor(SerialDenseExecutor):
    """Computes the spectral function in parallel over k and w using dense
    linear algebra."""

    def prime(self):

        if self.mpi_comm is None:
            self._logger.error("Prime failed, no MPI communicator provided")
            return

        self._dense_prime_helper()

    def spectrum(
        self, k, w, eta, return_G=False, return_meta=False, **solve_kwargs
    ):
        """Solves for the spectrum in parallel. Requires an initialized
        communicator at instantiation.

        Parameters
        ----------
        k : float or array_like
            The momentum quantum number point of the calculation.
        w : float or array_like
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.
        return_G : bool
            If True, returns the Green's function as opposed to the spectral
            function.
        return_meta : bool
            If True, returns a tuple of the Green's function and the dictionary
            containing meta information. If False, returns just the Green's
            function (the default is False).

        Returns
        -------
        np.ndarray
            The resultant spectrum.
        """

        k = float_to_list(k)
        w = float_to_list(w)

        # Generate a list of tuples for the (k, w) points to calculate.
        jobs = [(_k, _w) for _k in k for _w in w]

        # Chunk the jobs appropriately. Each of these lists look like the jobs
        # list above.
        jobs_on_rank = self.get_jobs_on_this_rank(jobs)
        self._logger.debug(f"{len(jobs_on_rank)} jobs todo")
        self._log_job_distribution_information(jobs_on_rank)
        self._total_jobs_on_this_rank = len(jobs_on_rank)

        # Get the results on this rank.
        s = [
            self.solve(_k, _w, eta, ii, **solve_kwargs)
            for ii, (_k, _w) in enumerate(jobs_on_rank)
        ]

        # Gather the results on rank 0
        all_results = self.mpi_comm.gather(s, root=0)

        if self.mpi_rank == 0:

            s = [xx[0] for xx in all_results]
            meta = [xx[1] for xx in all_results]
            if return_G:
                res = np.array(s)
            else:
                res = -np.array(s).imag / np.pi

            # Ensure the returned array has the proper shape
            res = res.reshape(len(k), len(w))
            if return_meta:
                return (res, meta)
            return res

    def dispersion(
        self, kgrid, w0, eta, eta_div=3.0, eta_step_div=5.0,
        next_k_offset_factor=1.5, nmax=1000
    ):
        """For now the dispersion method has to be run serially across (k,w)
        points and does not work with parallelization across (k,w). It does
        however work with the PETSc "across matrix" parallelization."""

        raise NotImplementedError
