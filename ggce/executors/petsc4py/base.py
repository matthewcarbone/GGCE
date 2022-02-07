#!/usr/bin/env python3

import numpy as np
import time
import os
import pickle

from petsc4py import PETSc

from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega
from ggce.utils.logger import Logger
from ggce.utils.utils import peak_location_and_weight, chunk_jobs, \
    float_to_list

BYTES_TO_GB = 1073741274

class BaseExecutorPETSC(SerialSparseExecutor):
    """A base class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel. This is built on top of the SerialSparseExecutor.
    This base class has fundamental methods such as matrix construction.
    The solve methods, as well as convergence and
    memory tracking are implemented in the inherited classes."""

    def __init__(
        self, model, default_console_logging_level='INFO',
        log_file=None, mpi_comm=None, brigade_size=1, log_every=1
    ):
        # Initialize the executor's logger and adjust the default logging level
        # for the console output
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_brigade = 1
        self.brigade_size = brigade_size
        self.mpi_world_size = 1
        if mpi_comm is not None:
            self.mpi_comm = mpi_comm
            self.mpi_rank = mpi_comm.Get_rank()
            self.mpi_world_size = mpi_comm.Get_size()
            self.brigades = int( self.mpi_world_size / self.brigade_size )
            self.mpi_brigade = int( self.mpi_rank / self.brigade_size )
        self._logger = Logger(log_file, mpi_rank=self.mpi_rank)
        self._logger.adjust_logging_level(default_console_logging_level)
        self._model = model
        self._system = None
        self._basis = None
        self._log_every = log_every
        self._total_jobs_on_this_brigade = 1

        ## for now the implementation has limitations: must have worldsize evenly divided into brigades
        try:
            assert (1-self.mpi_world_size % self.brigade_size)
        except AssertionError:
            self._logger.error(f"Number of MPI ranks cannot be equally divided into brigades. Exiting.")
            exit()

        if mpi_comm is not None:
            self.split_into_brigades()
            self.mpi_rank = self.mpi_comm_brigadier.Get_rank()


    def split_into_brigades(self):

        if self.brigades > 1:
            self.mpi_comm_brigadier = self.mpi_comm.Split(self.mpi_brigade,\
                                                            self.mpi_rank)
        else:
            self._logger.warning("Only one brigade, no splitting required. "\
                                                    "Using original MPI_COMM.")
            self.mpi_comm_brigadier = self.mpi_comm

    def get_jobs_on_this_brigade(self, jobs):
        """Get's the jobs assigned to this group of ranks. Note this method
        silently behaves as it should when the world size is 1, and will log
        a warning if it is called but the communicator is not initialized.

        Parameters
        ----------
        jobs : list
            The jobs to chunk

        Returns
        -------
        list
            The jobs assigned to this rank.
        """

        if self.mpi_comm_brigadier is None:
            self._logger.warning("Chunking jobs with COMM_WORLD_SIZE=1")
            return jobs

        return chunk_jobs(jobs, self.brigades, self.mpi_brigade)

    def set_input_dir(self, dir):

        self.basis_dir = dir

    def _setup_petsc_structs(self):
        """This function serves to initialize the various vectors and matrices
        (using PETSc data types) that are needed to solve the linear problem.
        They are setup using the sparse scheme, in parallel, so that each
        process owns only a small chunk of it."""

        # Initialize the parallel vector b from Ax = b
        self._vector_b = PETSc.Vec().create(comm=self.mpi_comm_brigadier)

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
        # self._logger.debug(f"I am rank {self.mpi_rank} in brigade {self.mpi_brigade} and got range {self._rstart} to {self._rend}")

        # Create the matrix for the linear problem
        self._mat_X = PETSc.Mat().create(comm=self.mpi_comm_brigadier)

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

        ## needed so that sole method does not fail
        self.basis_dir = None

        if self.mpi_comm_brigadier is None:
            self._logger.error("Prime failed, no MPI communicator provided")
            return

        self._sparse_prime_helper()

        # Get the total size of the linear system -- needed by PETSc
        self._linsys_size = len(self._basis)

        # Call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

    def prime_from_disk(self):
        """Prepare the executor for running by loading the system of equations
        from disk. Requires a communicator be provided at instantiation."""

        if self.mpi_comm_brigadier is None:
            self._logger.error("Prime failed, no MPI communicator provided")
            return

        self._logger.info(f"Matrices are loaded from disk. "\
                            f"We will not re-compute the basis.")

        # Get the total size of the linear system -- needed by PETSc
        assert self.basis_dir is not None
        self._linsys_size = self._get_matr_size(self.basis_dir)

        # Call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

    # @profile
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

        # quickly report the sparsity of the matrix
        self._lengthdat = len(dat)
        self._sparsity = (self._linsys_size**2 - len(dat)) / self._linsys_size**2
        self._edge_sparsity = len(dat) / self._linsys_size

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

    def _matrix_from_disk(self, k, w, eta, basis_dir):
        """The function uses the GGCE equation sparse format data to construct
        a sparse matrix in the PETSc scheme. Instead of using the basis,
        it loads the CSR elements from disk. The passed parameters
        are used to load the correct file from disk.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float
            The artificial broadening parameter of the calculation.

        Returns
        -------
        The matrices self._mat_X, self._vector_b are constructed in-place,
        nothing is returned.
        """

        matrix_loc = os.path.join(basis_dir, f"k_{k}_w_{w}_e_{eta}.bss")
        with open(matrix_loc, "rb") as datafile:
            row_ind, col_ind, dat = pickle.load(datafile)


        # quickly report the sparsity of the matrix
        self._lengthdat = len(dat)
        self._sparsity = (self._linsys_size**2 - len(dat)) / self._linsys_size**2
        self._edge_sparsity = len(dat) / self._linsys_size
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

    def _get_matr_size(self, matr_dir):

        """For use with the _matrix_from_disk method. Helps figure
           out the ultimate matrix size before loading all in."""

        all_files = os.listdir(matr_dir)
        all_files = [elem for elem in all_files if ".bss" in elem]
        random_matr = np.random.choice(all_files)
        sample_matrix = os.path.join(matr_dir, random_matr)
        with open(sample_matrix, "rb") as datafile:
            row_ind, col_ind, dat = pickle.load(datafile)

        matrsize = max(row_ind) + 1

        return matrsize

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

        # create variable measuring how much tolerance is met / exceeded
        # if positive, we are in trouble
        self.tol_excess = _vector_res_norm - rtol * _vector_b_norm

        # do a manual residual check on head node
        if self.mpi_rank == 0:
            if self.tol_excess > 0:
                self._logger.warning(
                    f"Rank {self.mpi_rank} in brigade {self.mpi_brigade} Solution failed residual relative tolerance check. "
                    "Solutions likely not fully converged: "
                    f"res_norm ({_vector_res_norm:.02e}) > "
                    f"rtol * b_norm ({rtol*_vector_b_norm:.02e})"
                )
            else:
                self._logger.debug("Solution passed manual residual check.")

    def spectrum(
        self, k, w, eta, return_G=False, return_meta=False, **solve_kwargs\
        ):
        """Solves for the spectrum using the PETSc solver backend. Computation
        is serial over k,w, but for each k,w it is massively paralle.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float, ndarray
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

        s = np.array([[
            self.solve(_k, _w, eta, **solve_kwargs)
            for ii, _w in enumerate(w)
        ] for jj, _k in enumerate(k)])

        # Separate meta information
        s_vals = s[:,:,0]
        meta = s[:,:,1]

        if return_G:
            return_vals = s_vals
        else:
            return_vals = [[-s_val.imag / np.pi for s_val in s_vals_array]\
                                                    for s_vals_array in s_vals]

        if return_meta:
            return (return_vals, meta)

        return return_vals
