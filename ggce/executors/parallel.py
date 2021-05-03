#!/usr/bin/env python3

import numpy as np
import time

from petsc4py import PETSc

from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega


class ParallelSparseExecutor(SerialSparseExecutor):
    """A class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel."""

    def _setup_petsc_structs(self):
        """This function serves to initialize the various vectors and matrices
        (using PETSc data types) that are needed to solve the linear problem.
        They are setup using the sparse scheme, in parallel, so that each
        process owns only a small chunk of it."""

        # Initialize the parallel vector b from Ax = b
        self._vector_b = PETSc.Vec().create(comm=self.comm)

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
        self._mat_X = PETSc.Mat().create(comm=self.comm)

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

        if self.comm is None:
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
        finfo = self._parameters.get_fFunctionInfo()
        G0 = G0_k_omega(k, w, finfo.a, finfo.eta, finfo.t)
        self._vector_b.setValues(self._linsys_size - 1, G0)

        # Need to assemble before use
        self._vector_b.assemblyBegin()
        self._vector_b.assemblyEnd()

        # TODO: check memory usage

        dt = time.time() - t0
        self._logger.debug("PETSc matrix assembled", elapsed=dt)

    def solve(self, k, w, eta=None, rtol=1.0e-15):
        """Solve the sparse-represented system using PETSc's KSP context.

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

        self.comm.barrier()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry
        if self.rank == self.comm.getSize() - 1:
            G = self._vector_x.getValue(self._linsys_size - 1)
            return np.array(G), {'time': [dt]}

        # TODO: figure out how to gather the vector into serial form using
        # the MPI communicator. I.e., return the entire resultant vector on
        # rank 0, not just G.
