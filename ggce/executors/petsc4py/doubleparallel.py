import numpy as np
import time

from petsc4py import PETSc

from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS
from ggce.utils.utils import peak_location_and_weight, chunk_jobs, \
    float_to_list

BYTES_TO_GB = 1073741274


class DoubleParallelExecutorMUMPS(ParallelSparseExecutorMUMPS):
    """A class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel, using a one-shot sparse sovler MUMPS.
    This inherits the matrix construction strategies of the BaseExecutorPETSC
    base class.

    This is done because e.g. convergence and memory checks are often specific
    to the particular solver used -- and so is the KSP context (i.e. solver)
    setup."""

    def check_conv(self, factored_mat, rtol, elapsed):
        """This helper function checks MUMPS convergence using built-in MUMPS
        error codes. Factored_mat holds the MUMPS error codes and control
        values.

        Parameters
        ----------
        factored_mat : PETSc_Mat
            The factorized linear system matrix from the PETSc'
            pre-condictioning context PC, obtained afetr the
            solver has been called.

        Returns
        -------
        The MUMPS convergence criterion is conducted in place, nothing is
        returned.
        """

        # MUMPS main convergence index -- if 0, all good
        self.mumps_conv_ind = factored_mat.getMumpsInfog(1)

        # do the MUMPS check on the head node
        if self.mpi_rank == 0:
            if self.mumps_conv_ind == 0:
                self._logger.debug(
                    "According to MUMPS diagnostics, call to MUMPS was "
                    f"successful. The calculation took {elapsed:.2f} sec."
                )
            elif self.mumps_conv_ind < 0:
                self._logger.error(
                    "A MUMPS error occured with MUMPS error code "
                    f"{self.mumps_conv_ind} See the MUMPS User Guide, Sec. 8, "
                    "for error  diagnostics. The calculation took "
                    f"{elapsed:.2f} sec."
                )
            elif self.mumps_conv_ind > 0:
                self._logger.warning(
                    "A MUMPS warning occured with MUMPS warning code "
                    f"{self.mumps_conv_ind} See the MUMPS User Guide, Sec. 8, "
                    "for error diagnostics. The calculation took "
                    f"{elapsed:.2f} sec."
                )

    def check_mem_use(self, factored_mat):
        """This helper function checks MUMPS memory usage with built-in MUMPS
        access.

        Parameters
        ----------
        factored_mat : PETSc_Mat
            The factorized linear system matrix from the PETSc'
            pre-condictioning context PC, obtained afetr the
            solver has been called.

        Returns
        -------
        The memory resuts are given to the logger, nothing is returned.
        """

        # Each rank reports their memory usage (in millions of bytes)
        self.rank_mem_used = factored_mat.getMumpsInfo(26) * 1e6 / BYTES_TO_GB
        self._logger.debug(
            f"Current rank MUMPS memory usage is {self.rank_mem_used:.02f} GB"
        )

        # set up memory usage tracking, report to the logger on head node only
        # total memory across all processes
        self.total_mem_used = factored_mat.getMumpsInfog(31) * 1e6 \
            / BYTES_TO_GB
        if self.mpi_rank == 0:
            self._logger.debug(
                f"Total MUMPS memory usage is {self.total_mem_used:.02f} GB"
            )

    # @profile
    def solve(self, k, w, eta, rtol=1.0e-10, **kwargs):
        """Solve the sparse-represented system using PETSc's KSP context.
        Note that this method only returns values on MPI rank = 0. All other
        ranks will return None.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.
        rtol : float, optional
            PETSc's relative tolerance (the default is 1.0e-10).

        Returns
        -------
        np.ndarray, dict
            The value of G and meta information, which in this case, is only
            specifically the time elapsed to solve for this (k, w) point
            using the PETSc KSP context.
        """

        # Function call to construct the sparse matrix into self._mat_X
        if self.basis_dir is None:
            self._assemble_matrix(k, w, eta)
        else:
            self._matrix_from_disk(k, w, eta, basis_dir = self.basis_dir)

        t0 = time.time()

        # Now construct the desired solver instance
        ksp = PETSc.KSP().create()

        # "preonly" for e.g. mumps and other external solvers
        ksp.setType('preonly')

        # Define the linear system matrix and its preconditioner
        ksp.setOperators(self._mat_X, self._mat_X)

        # Set preconditioner options (see PETSc manual for details)
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        # Set tolerance and options
        ksp.setTolerances(rtol=rtol)
        ksp.setFromOptions()

        dt = time.time() - t0
        self._logger.debug("KSP and PC contexts initialized", elapsed=dt)

        # Call the solve method
        t0 = time.time()
        ksp.solve(self._vector_b, self._vector_x)
        dt = time.time() - t0

        # assemble the solution vector
        self._vector_x.assemblyBegin()
        self._vector_x.assemblyEnd()

        self.mpi_comm_brigadier.barrier()

        # call manual residual check, as well as check MUMPS INFO
        # if the MUMPS solver call was successful
        # to get access to MUMPS error codes, get the factored matrix from PC
        factored_mat = pc.getFactorMatrix()
        self.check_conv_manual(pc, rtol=rtol)
        self.check_conv(factored_mat, rtol=rtol, elapsed=dt)

        # now check memory usage
        self.check_mem_use(factored_mat)

        self.mpi_comm_brigadier.barrier()

        # for memory management, destroy the KSP context manually
        ksp.destroy()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry aka "the last equation" of the matrix
        # use a gather operation, called by all ranks, to construct the full
        # vector (currently not used but will be later)
        G_vec = self.mpi_comm_brigadier.gather(self._vector_x.getArray(), root=0)

        # since we grabbed the Green's func value, destroy the data structs
        # self._vector_x.destroy()
        # self._vector_b.destroy()
        # self._mat_X.destroy()

        # Now select only the final value from the array
        if self.mpi_rank == 0:
            G_val = G_vec[self.brigade_size-1][-1]
        else:
            G_val = None

        # and bcast to all processes
        G_val = self.mpi_comm_brigadier.bcast(G_val, root=0)
        # self._logger.debug(f"We solved {k:.2f} {w:.3f}")
        # return np.array(G_val).round(decimals=3), {self.mpi_brigade: self.mpi_rank}

        return np.array(G_val), {
            'time': [dt],
            'mumps_exit_code': [self.mumps_conv_ind],
            'mumps_mem_tot': [self.total_mem_used],
            'manual_tolerance_excess': [self.tol_excess]
        }

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
        ## there are limitations: the number of jobs has to be evenly divisible by all the brigades
        try:
            assert (1-len(jobs) % self.brigades)
        except AssertionError:
            self._logger.error(f"Jobs ({len(jobs)}) cannot be equally divided"
                            f" between brigades ({self.brigades}). Exiting.")
            exit()

        # Chunk the jobs appropriately. Each of these lists look like the jobs
        # list above.
        jobs_on_brigade = self.get_jobs_on_this_brigade(jobs)
        self._total_jobs_on_this_brigade = len(jobs_on_brigade)

        # Get the results on this rank.
        s = [
            list(self.solve(_k, _w, eta, **solve_kwargs))
            for ii, (_k, _w) in enumerate(jobs_on_brigade)
        ]

        # Gather the results from the sergeants to "the general" (global rank 0)
        all_results = self.mpi_comm.gather(s, root=0)

        ## need to get rid of duplicates, since each rank in a brigade sends
        if self.mpi_rank == 0 and self.mpi_brigade == 0:
            results = []
            if self.brigade_size > 1:
                for n in range(self.brigades):
                    results.append(all_results[int(n*self.brigade_size)])
            else:
                results = all_results

            results = [xx[ii] for xx in results for ii in range(len(xx))]

            ## a copy of the results of the whole brigade
            s = [xx[0] for xx in results]
            meta = [xx[1] for xx in results]
            if return_G:
                res = np.array(s)
            else:
                res = -np.array(s).imag / np.pi

            # Ensure the returned array has the proper shape
            res = res.reshape(len(k), len(w))
            if return_meta:
                return (res, meta)
            return res
