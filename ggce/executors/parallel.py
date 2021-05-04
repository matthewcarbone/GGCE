#!/usr/bin/env python3

import numpy as np

from ggce.executors.serial import SerialDenseExecutor


class ParallelDenseExecutor(SerialDenseExecutor):
    """Computes the spectral function in parallel over k and w using dense
    linear algebra."""

    def prime(self):

        if self.mpi_comm is None:
            self._logger.error("Prime failed, no MPI communicator provided")
            return

        self._dense_prime_helper()

    def spectrum(self, k, w, eta=None):
        """Solves for the spectrum in parallel. Requires an initialized
        communicator at instantiation.

        Parameters
        ----------
        k : float or array_like
            The momentum quantum number point of the calculation.
        w : float or array_like
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

        if isinstance(k, (float, int)):
            k = [k]
        if isinstance(w, (float, int)):
            w = [w]

        # Generate a list of tuples for the (k, w) points to calculate.
        jobs = [(_k, _w) for _w in w for _k in k]

        # Chunk the jobs appropriately. Each of these lists look like the jobs
        # list above.
        jobs_on_rank = self.get_jobs_on_this_rank(jobs)

        # Get the results on this rank.
        s = [-self.solve(_k, _w)[0].imag / np.pi for (_k, _w) in jobs_on_rank]

        # Gather the results on rank 0
        all_results = self.comm.gather(s, root=0)

        if self.rank == 0:
            return np.array(all_results).reshape(len(k), len(w))
