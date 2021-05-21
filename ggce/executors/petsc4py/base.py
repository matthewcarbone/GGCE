#!/usr/bin/env python3

import numpy as np
import time

from petsc4py import PETSc

from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega

BYTES_TO_GB = 1073741274

class BaseExecutorPETSC(SerialSparseExecutor):
    """A base class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel. This is built on top of the SerialSparseExecutor.
    This base class has fundamental methods such as matrix construction.
    The solve methods, as well as convergence and
    memory tracking are implemented in the inherited classes."""

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

        ## parse out the nonzero (nnz) matrix structure across rows
        ## so we can pre-allocate enough space for the matrix
        ## avoid wasting space and speed up assembly ~ 20x

        ## set up arrays of length equal to space owned by a given MPI process
        ## diag and offdiag store the number of nonzero entries in a given row
        ## in the diagonal or off-diagonal block of the matrix
        diag_nnz = np.zeros(self._rend-self._rstart, dtype='i4')
        offdiag_nnz = np.zeros(self._rend-self._rstart, dtype='i4')

        ## iterate through coo notation arrays to identify the nonzero entry
        ## number in each row
        for i, elem in enumerate(row_ind):
            # check if this row / column is owned by this MPI process
            if self._rstart <= elem and elem < self._rend:
                if self._rstart <= col_ind[i] and col_ind[i] < self._rend:
                    diag_nnz[elem-self._rstart] += 1
                else:
                    offdiag_nnz[elem-self._rstart] += 1

        ## pass the nnz arrays to PETSC matrix
        self._mat_X.setPreallocationNNZ((diag_nnz,offdiag_nnz))

        ## now populate the matrix with actual values
        row_start = np.zeros(1, dtype='i4')
        col_pos = np.zeros(1, dtype='i4')
        val = np.zeros(1, dtype='complex128')
        for ii, row_coo in enumerate(row_ind):
            if self._rstart <= row_coo and row_coo < self._rend:
                row_start, col_pos, val = row_coo, col_ind[ii], dat[ii]
                # self._logger.debug(f"I am rank {self.mpi_rank} and I am setting the values at {(row_start, col_pos)}")
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

        ## TODO: check memory usage
        ## presently not wrapped for Python

        dt = time.time() - t0
        self._logger.debug("PETSc matrix assembled", elapsed=dt)

    def check_conv_manual(self, pc, rtol):
        """This helper function checks PETSC convergence manually, by computing
        the residual r = b - Ax directly, with the preconditioners applied,
        and comparing it to the rtol*||b||_2 convergence criterion.

        Parameters
        ----------
        pc           : PETSc_PC
            This is the preconditioner context from PETSc. In particular
            it allows us to manually compute the residual by applying
            the preconditioner to the residual vector because we take the
            norm. With left preconditioning, which is the default, residual
            norm is taken with preconditioner applied on the left.

        Returns
        -------
        The residual check is conducted in place, nothing is returned.
        """

        # compute the residual and apply the preconditioner
        _vector_res = self._vector_b.copy()
        pc.apply(self._vector_b - self._mat_X(self._vector_x), _vector_res)
        _vector_res_norm = _vector_res.norm(PETSc.NormType.NORM_2)
        # tolerance comparison is based on rtol * b magnitude, which also needs
        # to be preconditioned
        _vector_b_condt = self._vector_b.duplicate()
        pc.apply(self._vector_b, _vector_b_condt)
        _vector_b_norm  = _vector_b_condt.norm(PETSc.NormType.NORM_2)

        # do a manual residual check on head node
        if self.mpi_rank == 0:
            if _vector_res_norm > (rtol * _vector_b_norm):
                self._logger.warning(
                    "Solution failed residual relative tolerance check. "
                    "Solutions likely not fully converged: "
                    f"res_norm ({_vector_res_norm:.02e}) > "
                    f"rtol * b_norm ({rtol*_vector_b_norm:.02e})"
                )
            else:
                self._logger.debug("Solution passed manual residual check.")
