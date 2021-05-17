#!/usr/bin/env python3

import numpy as np
import time

from petsc4py import PETSc

from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega

BYTES_TO_GB = 1073741274


class ParallelSparseExecutor(SerialSparseExecutor):
    """A class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel."""

    def _setup_petsc_structs(self):
        """This function serves to initialize the various vectors and matrices
        (using PETSc data types) that are needed to solve the linear problem.
        They are setup using the sparse scheme, in parallel, so that each
        process owns only a small chunk of it."""

        # Initialize the parallel vector b from Ax = b
        self._vector_b = PETSc.Vec().create(comm=self.mpi_comm)

        # Need to set the total size of the vector
        self._vector_b.setSizes(self._linsys_size)

        # This sets all the other PETSc options as defaults
        self._vector_b.setFromOptions()

        # Now we create the solution vector, x in Ax = b
        self._vector_x = self._vector_b.duplicate()

        # Now determine what is the local size PETSc picked
        _n_local = self._vector_b.getLocalSize()

        # Figure out what the given process owns
        self._rstart, self._rend = self._vector_b.getOwnershipRange()

        # Create the matrix for the linear problem
        self._mat_X = PETSc.Mat().create(comm=self.mpi_comm)

        # set the matrix dimensions
        # input format is [(n,N),(m,M)] where capitals are total matrix
        # dimensions and lowercase are local block dimensions
        # see bottom of PETSc listserv entry
        # https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2015-March/024879.html
        # for example
        self._mat_X.setSizes(
            [(_n_local, self._linsys_size), (_n_local, self._linsys_size)]
        )
        # This sets all the other PETSc options as defaults
        self._mat_X.setFromOptions()

        # This is needed for some reason before PETSc matrix can be used
        self._mat_X.setUp()

    def prime(self):
        """Prepare the executor for running by finding the system of equations
        and basis. Requires a communicator be provided at instantiation."""

        if self.mpi_comm is None:
            self._logger.error("Prime failed, no MPI communicator provided")
            return

        self._sparse_prime_helper()

        # Get the total size of the linear system -- needed by PETSc
        self._linsys_size = len(self._basis)

        # Call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

    def _assemble_matrix(self, k, w, eta):
        """The function uses the GGCE equation sparse format data to construct
        a sparse matrix in the PETSc scheme.

        Parameters
        ----------
        k : float or array_like
            The momentum quantum number point of the calculation.
        w : float or array_like
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.

        Returns
        -------
        The matrices self._mat_X, self._vector_b are constructed in-place,
        nothing is returned.
        """

        row_ind, col_ind, dat = self._sparse_matrix_from_equations(k, w, eta)

        t0 = time.time()

        row_start = np.zeros(1, dtype='i4')
        col_pos = np.zeros(1, dtype='i4')
        val = np.zeros(1, dtype='complex128')
        loc_extent = range(self._rstart, self._rend)
        for ii, row_coo in enumerate(row_ind):
            if row_coo in loc_extent:
                row_start, col_pos, val = row_coo, col_ind[ii], dat[ii]
                self._mat_X.setValues(row_start, col_pos, val)

        # Assemble the matrix now that the values are filled in
        self._mat_X.assemblyBegin(self._mat_X.AssemblyType.FINAL)
        self._mat_X.assemblyEnd(self._mat_X.AssemblyType.FINAL)

        # Assign values for the b vector
        finfo = self._model.get_fFunctionInfo()
        G0 = G0_k_omega(k, w, finfo.a, eta, finfo.t)
        self._vector_b.setValues(self._linsys_size - 1, G0)

        # Need to assemble before use
        self._vector_b.assemblyBegin()
        self._vector_b.assemblyEnd()

        # TODO: check memory usage

        dt = time.time() - t0
        self._logger.debug("PETSc matrix assembled", elapsed=dt)

    def check_convergence(self, factored_mat, rtol):
        """This helper function checks MUMPS convergence using built-in MUMPS
        error codes and a manual residual check.

        Parameters
        ----------
        factored_mat : PETSc_Mat
            The factorized linear system matrix from the PETSc'
            pre-condictioning context PC, obtained afetr the
            solver has been called.

        Returns
        -------
        The residual check and MUMPS convergenc criterions are conducted
        in place, nothing is returned.
        """

        # MUMPS main convergence index -- if 0, all good
        mumps_conv_ind = factored_mat.getMumpsInfog(1)
        if mumps_conv_ind == 0:
            self._logger.debug(
                "According to MUMPS diagnostics, call to MUMPS was "
                "successful."
            )
        elif mumps_conv_ind < 0:
            self._logger.error(
                "A MUMPS error occured with MUMPS error code "
                f"{mumps_conv_ind} See the MUMPS User Guide, Sec. 8, for "
                "error  diagnostics."
            )
        elif mumps_conv_ind > 0:
            self._logger.warning(
                "A MUMPS warning occured with MUMPS warning code "
                f"{mumps_conv_ind} See the MUMPS User Guide, Sec. 8, for "
                "error diagnostics."
            )

        # Unhappy with MUMPS, we do our own double-check of residual criterion
        res = self._vector_b - self._mat_X(self._vector_x)
        res_norm = res.norm(PETSc.NormType.NORM_2)
        if self.mpi_rank == 0:
            self._logger.debug(
                f"MUMPS final residual is {res_norm}, rtol is {rtol}"
            )

        if res_norm > rtol:
            self._logger.warning(
                "Solution failed residual relative tolerance check. "
                "Solutions likely not fully converged: "
                f"res_norm ({res_norm:.02e}) > rtol ({rtol:.02e})"
            )

        else:
            self._logger.debug("Solution passed manual residual check.")

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
        rank_mem_used = factored_mat.getMumpsInfo(26) * 1e6 / BYTES_TO_GB
        self._logger.debug(
            f"Current rank MUMPS memory usage is {rank_mem_used:.02f} GB"
        )

        # set up memory usage tracking, report to the logger
        if self.mpi_rank == 0:
            # total memory across all processes
            total_mem_used = factored_mat.getMumpsInfog(31) * 1e6 / BYTES_TO_GB
            self._logger.debug(
                f"Total MUMPS memory usage is {total_mem_used:.02f} GB"
            )

    def solve(self, k, w, eta, rtol=1.0e-15):
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
            PETSc's relative tolerance (the default is 1.0e-15).

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

        self._logger.debug("Sparse matrices solved", elapsed=dt)

        # assemble the solution vector
        self._vector_x.assemblyBegin()
        self._vector_x.assemblyEnd()

        self.mpi_comm.barrier()

        # implement manual residual check, as well as check MUMPS INFO
        # if the MUMPS solver call was successful
        # to get access to MUMPS error codes, get the factored matrix from PC
        factored_mat = pc.getFactorMatrix()
        self.check_convergence(factored_mat, rtol=rtol)

        # now check memory usage
        self.check_mem_use(factored_mat)

        self.mpi_comm.barrier()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry aka "the last equation" of the matrix
        # use a gather operation, called by all ranks, to construct the full
        # vector
        G = self.mpi_comm.gather(self._vector_x.getArray(), root=0)
        # this returns a list of values of G from all processes

        if self.mpi_rank == 0:
            # now select only the final process list and final value
            G = G[self.mpi_world_size-1][-1]
            return np.array(G), {'time': [dt]}
