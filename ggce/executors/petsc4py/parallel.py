import numpy as np
import time

from petsc4py import PETSc

from ggce.executors.petsc4py.base import BaseExecutorPETSC
from ggce.utils.utils import float_to_list

BYTES_TO_GB = 1073741274


class ParallelSparseExecutorMUMPS(BaseExecutorPETSC):
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

    def solve(self, k, w, eta, rtol=1.0e-10):
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
        self._assemble_matrix(k, w, eta)

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

        self.mpi_comm.barrier()

        # call manual residual check, as well as check MUMPS INFO
        # if the MUMPS solver call was successful
        # to get access to MUMPS error codes, get the factored matrix from PC
        factored_mat = pc.getFactorMatrix()
        self.check_conv_manual(pc, rtol=rtol)
        self.check_conv(factored_mat, rtol=rtol, elapsed=dt)

        # now check memory usage
        self.check_mem_use(factored_mat)

        self.mpi_comm.barrier()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry aka "the last equation" of the matrix
        # use a gather operation, called by all ranks, to construct the full
        # vector (currently not used but will be later)
        G_vec = self.mpi_comm.gather(self._vector_x.getArray(), root=0)

        # Now select only the final value from the array
        if self.mpi_rank == 0:
            G_val = G_vec[self.mpi_world_size-1][-1]
        else:
            G_val = None

        # and bcast to all processes
        G_val = self.mpi_comm.bcast(G_val, root=0)

        return np.array(G_val), {
            'time': [dt],
            'mumps_exit_code': [self.mumps_conv_ind],
            'mumps_mem_tot': [self.total_mem_used],
            'manual_tolerance_excess': [self.tol_excess]
        }


class ParallelSparseExecutorGMRES(BaseExecutorPETSC):
    """A class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel, using an iterative solver GMRES.
    This inherits the matrix construction strategies of the BaseExecutorPETSC
    base class.

    This is done because e.g. convergence and memory checks are often specific
    to the particular solver used -- and so is the KSP context (i.e. solver)
    setup."""

    def check_conv(self, ksp, rtol, elapsed):
        """This helper function checks GMRES convergence using built-in PETSc
        error codes.

        Parameters
        ----------
        ksp : PETSc_KSP
            The Krylov Subspace solution context that contains convergence
            codes, convergence history (i.e. residuals from iterations)
            and more.

        Returns
        -------
        The residual check and PETSc convergence criterions are conducted
        in place, nothing is returned.
        """

        # GMRES main convergence index
        gmres_conv_ind = ksp.getConvergedReason()
        if self.mpi_rank == 0:
            if gmres_conv_ind > 0:
                self._logger.debug(
                    "According to PETSc diagnostics, call to GMRES was "
                    f"successful. It exited with code {gmres_conv_ind}."
                    "See the PETSc header in petsc/include/petscksp.h, "
                    f"lines 518-680 for details. The calculation took "
                    f"{elapsed:.2f} sec."
                )
            elif gmres_conv_ind < 0:
                self._logger.error(
                    "A PETSc calculation divergence was detected with error "
                    f"code {gmres_conv_ind}. See include/petscksp.h, "
                    "lines 518-680 for error  diagnostics. The calculation "
                    f"took {elapsed:.2f} sec."
                )

        # now do a check using the final residual from getconvergenceHistory
        res_hist = ksp.getConvergenceHistory()
        its = ksp.getIterationNumber()
        # if on head node, log the residual history for debugging
        if self.mpi_rank == 0:
            self._logger.debug(
                f"Run ended after {its} iterations. "
                f"Convergence history is \n {res_hist}."
            )

    def check_mem_use(self, factored_mat):
        """This helper function checks PETSc sovler memory usage
        access.

        Parameters
        ----------
        factored_mat : PETSc_Mat
            The factorized linear system matrix from the PETSc'
            pre-condictioning context PC, obtained after the
            solver has been called.

        Returns
        -------
        The memory resuts are given to the logger, nothing is returned.

        WARNING: unfortunately PETSc python wrapper does not have wrapping for
        memory stuff.
        """

        raise NotImplementedError

    def solve(self, k, w, eta, index=None, rtol=1.0e-15):
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
        self._assemble_matrix(k, w, eta)

        t0 = time.time()

        # Now construct the desired solver instance
        ksp = PETSc.KSP().create()

        # setting type of the solver to GMRES
        ksp.setType('gmres')

        # Define the linear system matrix and its preconditioner
        ksp.setOperators(self._mat_X, self._mat_X)

        # Set preconditioner options (see PETSc manual for details)
        # is not required for GMRES, set by default to block Jacobi
        # we still extract PC context for preconditioner access
        pc = ksp.getPC()

        # Set tolerance and remaining options from command line (if any)
        ksp.setTolerances(rtol=rtol)
        ksp.setFromOptions()
        # Make a call to set up arrays for residual tracking
        ksp.setConvergenceHistory()

        dt = time.time() - t0
        self._logger.debug("KSP context initialized", elapsed=dt)

        # Call the solve method
        t0 = time.time()
        ksp.solve(self._vector_b, self._vector_x)
        dt = time.time() - t0

        ## export some data from the solver and return on debug
        # solver_info = ksp.view()
        # if self.mpi_rank == 0:
        #     self._logger.debug(solver_info)

        # assemble the solution vector
        self._vector_x.assemblyBegin()
        self._vector_x.assemblyEnd()

        self.mpi_comm.barrier()

        # Implement manual residual check, as well as check PETSc output code
        # in there we call for convergence history
        self.check_conv_manual(pc, rtol=rtol)
        self.check_conv(ksp, rtol=rtol, elapsed=dt)

        # Now check memory usage
        # memory check not wrapped in petsc4py for GMRES
        # self.check_mem_use(factored_mat)

        self.mpi_comm.barrier()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry aka "the last equation" of the matrix
        # use a gather operation, called by all ranks, to construct the full

        # vector
        G = self.mpi_comm.gather(self._vector_x.getArray(), root=0)

        if self.mpi_rank == 0:

            # Now select only the final process list and final value
            G = G[self.mpi_world_size - 1][-1]
            A = -G.imag / np.pi
            if A < 0.0:
                self._log_spectral_error(k, w)
            self._log_current_status(k, w, A, index, time.time() - t0)
            return np.array(G), {'time': [dt]}

    def spectrum(
        self, k, w, eta, return_G=False, return_meta=False, **solve_kwargs
    ):
        """Solves for the spectrum using the PETSc solver backend. Computation
        is serial over k,w, but for each k,w it is massively paralle.

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

        # All of the jobs run "on the same rank" in this context, whereas in
        # reality self.solve is parallel for every k,w point
        self._total_jobs_on_this_rank = len(k) * len(w)

        s = [[
            self.solve(_k, _w, eta, ii + jj * len(k), **solve_kwargs)
            for ii, _w in enumerate(w)
        ] for jj, _k in enumerate(k)]

        if self.mpi_rank == 0:

            # Separate meta information
            s = [xx[0] for xx in s]
            meta = [xx[1] for xx in s]
            if return_G:
                s = np.array(s)
            else:
                s = -np.array(s).imag / np.pi
            if return_meta:
                return (s, meta)
            return s
