import numpy as np
import time
import os
import pickle
from tqdm import tqdm

from petsc4py import PETSc

from ggce.logger import logger
from ggce.utils.physics import G0_k_omega
from ggce.utils.utils import chunk_jobs_equalize, float_to_list
from ggce.executors.solvers import Solver

BYTES_TO_GB = 1073741274


class MassSolver(Solver):
    """A base class to connect to PETSc's powerful parallel sparse solver
    tools, to calculate G(k,w) in parallel. This is an abstract base class
    built on top of the abstract Solver class. This base class has fundamental
    methods such as matrix construction. The solve methods, as well as
    convergence and memory tracking are implemented in the inherited classes,
    while some basic routines like spectrum() that are method-agnostic are
    implemented here."""

    @property
    def mpi_brigade(self):
        if self._brigade_size is not None:
            return int(self.mpi_rank / self._brigade_size)
        return 0

    @property
    def mpi_comm_brigadier(self):
        return self._mpi_comm_brigadier

    @property
    def brigade_size(self):
        if self._brigade_size is not None:
            return self._brigade_size
        return self._mpi_comm.Get_size()

    @property
    def brigades(self):
        if self._brigade_size is not None:
            return int(self._mpi_comm.Get_size() / self._brigade_size)
        return 1

    @property
    def brigade_rank(self):
        if self._brigade_size is not None:
            return self._mpi_comm_brigadier.Get_rank()
        return self.mpi_rank

    @property
    def matr_dir(self):
        """This property sets the directory where the method
        _scaffold_from_disk looks for pickled matrices (in CSR format).
        It is set in __init__
        """
        if self._matr_dir is None:
            logger.warning("matr_dir not set -- GGCE will construct matrices.")
        return self._matr_dir

    def __init__(self, brigade_size=None, matr_dir=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._matr_dir = matr_dir

        if self._mpi_comm is None:
            logger.critical(
                "PETSc solver cannot run with "
                "mpi_comm=None. Pass MPI_COMM when "
                "instantiating the MassSolver."
            )

        # brigade split
        self._brigade_size = brigade_size
        if self._brigade_size is not None:
            self.split_into_brigades()
        else:
            logger.warning(
                "Only one brigade, no splitting required. "
                "Using original MPI_COMM."
            )
            self._mpi_comm_brigadier = self._mpi_comm

    def split_into_brigades(self):
        """Splits the MPI_COMM provided into 'brigades' of ranks operating
        together. Does this on the basis of the provided _brigade_size,
        by assigning a process to a brigade on the basis of its global
        rank modulo the brigade size. mpi_brigade is automatically
        evaluated as a @property of the class.

        If the ranks cannot be evenly divided into brigades, raises an error
        and terminates the calculation. In future releases brigade splits
        will be able to handle non-even division and/or adjust on the fly.

        Returns
        -------
        None
            New mpi_comm_brigadier are set as attributes of the class.
        """

        # for now the implementation has limitations: must have worldsize
        # evenly divided into brigades
        assert 1 - self.mpi_world_size % self._brigade_size, logger.error(
                f"Number of MPI ranks {self.mpi_world_size} cannot be "\
                f"equally divided into brigades with size {self._brigade_size}."
            )

        self._mpi_comm_brigadier = self._mpi_comm.Split(
            self.mpi_brigade, self.mpi_rank
        )

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
            The jobs assigned to this brigade.
        """

        if self.brigades == 1:
            logger.warning("Chunking jobs with COMM_WORLD_SIZE=1")
            return jobs
        return chunk_jobs_equalize(jobs, self.brigades, self.mpi_brigade)

    def _setup_petsc_structs(self):
        """This function serves to initialize the various vectors and matrices
        (using PETSc data types) that are needed to solve the linear problem.
        They are set up using the sparse scheme, in parallel, so that each
        process owns only a small chunk of it.

        _mpi_comm_brigadier is used throughout to make sure that separate
        brigades work on separate (k,w) points, as is intended by the double-
        parallelization scheme.
        """

        # Initialize the parallel vector b from Ax = b
        self._vector_b = PETSc.Vec().create(comm=self._mpi_comm_brigadier)

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
        logger.debug(
            f"I am rank {self.mpi_rank} in brigade "\
            f"{self.mpi_brigade} and got range "\
            f"{self._rstart} to {self._rend}"
        )

        # Create the matrix for the linear problem
        self._mat_X = PETSc.Mat().create(comm=self._mpi_comm_brigadier)

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

        # This actually creates the matrix
        self._mat_X.setUp()

    def _sparse_matrix_from_equations(self, k, w, eta):
        """This function iterates through the GGCE equations dicts to extract
        the row, column coordiante and value of the nonzero entries in the
        matrix. This is subsequently used to construct the parallel sparse
        system matrix. This is exactly the same as in the Serial class: however
        that method returns X, v whereas here we need row_ind/col_ind_dat.

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
        list, list, list
            The row and column coordinate lists, as well as a list of values of
            the matrix that are nonzero.
        """

        row_ind = []
        col_ind = []
        dat = []

        total_bosons = np.sum(self._system.model.phonon_number)
        for n_bosons in range(total_bosons + 1):
            for eq in self._system.equations[n_bosons]:
                row_dict = dict()
                index_term_id = eq.index_term.id()
                ii_basis = self._basis[index_term_id]

                for term in eq._terms_list + [eq.index_term]:
                    jj = self._basis[term.id()]
                    try:
                        row_dict[jj] += term.coefficient(k, w, eta)
                    except KeyError:
                        row_dict[jj] = term.coefficient(k, w, eta)

                row_ind.extend([ii_basis for _ in range(len(row_dict))])
                col_ind.extend([key for key, _ in row_dict.items()])
                dat.extend([value for _, value in row_dict.items()])

        # estimate sparse matrix memory usage
        # (complex (16 bytes) + int (4 bytes) + int) * nonzero entries
        est_mem_used = 24 * len(dat) / BYTES_TO_GB
        logger.debug(f"Estimated memory needed is {est_mem_used:.02f} MB")

        return row_ind, col_ind, dat

    def _scaffold(self, k, w, eta):
        """This function uses the GGCE equation sparse format data to construct
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

        self._linsys_size = len(self._basis)

        row_ind, col_ind, dat = self._sparse_matrix_from_equations(k, w, eta)

        # quickly report the sparsity of the matrix
        self._lengthdat = len(dat)
        self._sparsity = (
            self._linsys_size**2 - len(dat)
        ) / self._linsys_size**2
        self._edge_sparsity = len(dat) / self._linsys_size

        t0 = time.time()

        # parse out the nonzero (nnz) matrix structure across rows
        # so we can pre-allocate enough space for the matrix
        # avoid wasting space and speed up assembly ~ 20x

        # Call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

        # set up arrays of length equal to space owned by a given MPI process
        # diag and offdiag store the number of nonzero entries in a given row
        # in the diagonal or off-diagonal block of the matrix
        diag_nnz = np.zeros(self._rend - self._rstart, dtype="i4")
        offdiag_nnz = np.zeros(self._rend - self._rstart, dtype="i4")

        # iterate through coo notation arrays to identify the nonzero entry
        # number in each row
        for i, elem in enumerate(row_ind):
            # check if this row / column is owned by this MPI process
            if self._rstart <= elem and elem < self._rend:
                if self._rstart <= col_ind[i] and col_ind[i] < self._rend:
                    diag_nnz[elem - self._rstart] += 1
                else:
                    offdiag_nnz[elem - self._rstart] += 1

        # pass the nnz arrays to PETSC matrix
        self._mat_X.setPreallocationNNZ((diag_nnz, offdiag_nnz))

        # now populate the matrix with actual values
        row_start = np.zeros(1, dtype="i4")
        col_pos = np.zeros(1, dtype="i4")
        val = np.zeros(1, dtype="complex128")
        for ii, row_coo in enumerate(row_ind):
            if self._rstart <= row_coo and row_coo < self._rend:
                row_start, col_pos, val = row_coo, col_ind[ii], dat[ii]
                self._mat_X.setValues(row_start, col_pos, val)

        # Assemble the matrix now that the values are filled in
        self._mat_X.assemblyBegin(self._mat_X.AssemblyType.FINAL)
        self._mat_X.assemblyEnd(self._mat_X.AssemblyType.FINAL)

        # Assign values for the b vector
        a = self._system.model.lattice_constant
        t = self._system.model.hopping
        G0 = G0_k_omega(k, w, a, eta, t)
        self._vector_b.setValues(self._linsys_size - 1, G0)

        # Need to assemble before use
        self._vector_b.assemblyBegin()
        self._vector_b.assemblyEnd()

        # TODO: check memory usage
        # presently not wrapped for Python

        dt = time.time() - t0
        logger.debug("PETSc matrix assembled", elapsed=dt)

    def _scaffold_from_disk(self, k, w, eta, matr_dir):
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
        matr_dir : string (path)
            The absolute path of the location of the matrices to be loaded.

        Returns
        -------
        The matrices self._mat_X, self._vector_b are constructed in-place,
        nothing is returned.
        """

        logger.info(
            "Matrices are loaded from disk. GGCE will not re-compute them."
        )

        # Get the total size of the linear system -- needed by PETSc
        assert matr_dir is not None
        self._linsys_size = self._get_matr_size(matr_dir)

        matrix_loc = os.path.join(matr_dir, f"k_{k}_w_{w}_e_{eta}.bss")
        with open(matrix_loc, "rb") as datafile:
            row_ind, col_ind, dat = pickle.load(datafile)

        # quickly report the sparsity of the matrix
        self._lengthdat = len(dat)
        self._sparsity = (
            self._linsys_size**2 - len(dat)
        ) / self._linsys_size**2
        self._edge_sparsity = len(dat) / self._linsys_size
        t0 = time.time()

        # parse out the nonzero (nnz) matrix structure across rows
        # so we can pre-allocate enough space for the matrix
        # avoid wasting space and speed up assembly ~ 20x

        # Call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

        # set up arrays of length equal to space owned by a given MPI process
        # diag and offdiag store the number of nonzero entries in a given row
        # in the diagonal or off-diagonal block of the matrix
        diag_nnz = np.zeros(self._rend - self._rstart, dtype="i4")
        offdiag_nnz = np.zeros(self._rend - self._rstart, dtype="i4")

        # iterate through coo notation arrays to identify the nonzero entry
        # number in each row
        for i, elem in enumerate(row_ind):
            # check if this row / column is owned by this MPI process
            if self._rstart <= elem and elem < self._rend:
                if self._rstart <= col_ind[i] and col_ind[i] < self._rend:
                    diag_nnz[elem - self._rstart] += 1
                else:
                    offdiag_nnz[elem - self._rstart] += 1

        # pass the nnz arrays to PETSC matrix
        self._mat_X.setPreallocationNNZ((diag_nnz, offdiag_nnz))

        # now populate the matrix with actual values
        row_start = np.zeros(1, dtype="i4")
        col_pos = np.zeros(1, dtype="i4")
        val = np.zeros(1, dtype="complex128")
        for ii, row_coo in enumerate(row_ind):
            if self._rstart <= row_coo and row_coo < self._rend:
                row_start, col_pos, val = row_coo, col_ind[ii], dat[ii]
                logger.debug(f"I am rank {self.mpi_rank} and I am setting"\
                f" the values at {(row_start, col_pos)}")
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
        # presently not wrapped for Python

        dt = time.time() - t0
        logger.debug(
            f"PETSc matrix assembled, built from disk at: {self._matr_dir}",
            elapsed=dt,
        )

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
        _vector_b_norm = _vector_b_condt.norm(PETSc.NormType.NORM_2)

        # create variable measuring how much tolerance is met / exceeded
        # if positive, we are in trouble
        self.tol_excess = _vector_res_norm - rtol * _vector_b_norm

        # do a manual residual check on head node
        if self.mpi_rank == 0:
            if self.tol_excess > 0:
                logger.warning(
                    f"Solution failed residual relative tolerance check. "
                    "Solutions likely not fully converged: "
                    f"res_norm ({_vector_res_norm:.02e}) > "
                    f"rtol * b_norm ({rtol*_vector_b_norm:.02e})"
                )
            else:
                logger.debug("Solution passed manual residual check.")

    def spectrum(self, k, w, eta, return_meta=False, pbar=False):
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
        return_meta : bool
            If True, returns a tuple of the Green's function and the dictionary
            containing meta information. If False, returns just the Green's
            function (the default is False).

        Returns
        -------
        np.ndarray
            The resultant Green's function array of shape nk by nw.
        """

        k = float_to_list(k)
        w = float_to_list(w)

        # Generate a list of tuples for the (k, w) points to calculate.
        jobs = [(_k, _w) for _k in k for _w in w]

        # Chunk the jobs appropriately. Each of these lists look like the jobs
        # list above.
        jobs_on_brigade = self.get_jobs_on_this_brigade(jobs)
        self._total_jobs_on_this_brigade = len(jobs_on_brigade)
        # Get the results on this rank.
        s = []
        for (_k, _w) in tqdm(jobs_on_brigade, disable=not pbar):
            s.append(self.solve(_k, _w, eta))

        # Gather the results from the brigade commanders to "the general"
        # (global rank 0)
        all_results = self._mpi_comm.gather(s, root=0)
        # create placeholder variables for final bcast of results
        res = None
        meta = None

        # need to get rid of duplicates, since each rank in a brigade sends
        # a copy of the results from the brigade
        if self.mpi_rank == 0:
            results = []
            if self.brigade_size > 1:
                for n in range(self.brigades):
                    results.append(all_results[int(n * self.brigade_size)])
            else:
                results = all_results

            results = [xx[ii] for xx in results for ii in range(len(xx))]

            # a copy of the results of the whole brigade
            s = [xx[0] for xx in results]
            meta = [xx[1] for xx in results]
            res = np.array(s)

            # remove nans from the padding
            res = res[~np.isnan(res)]

            # Ensure the returned array has the proper shape
            res = res.reshape(len(k), len(w))

        # Broadcast the final result to all ranks
        # memory intensive but intuitive
        res = self._mpi_comm.bcast(res, root=0)
        meta = self._mpi_comm.bcast(meta, root=0)

        if return_meta:
            return (res, meta)
        return res

    @staticmethod
    def _k_omega_eta_to_str(k, omega, eta):
        # Note this will have to be redone when k is a vector in 2 and 3D!
        return f"{k:.10f}_{omega:.10f}_{eta:.10f}"

    @staticmethod
    def _get_matr_size(self, matr_dir):

        """For use with the _scaffold_from_disk method. Helps figure
        out the ultimate matrix size before loading all in."""

        all_files = os.listdir(matr_dir)
        all_files = [elem for elem in all_files if ".bss" in elem]
        random_matr = np.random.choice(all_files)
        sample_matrix = os.path.join(matr_dir, random_matr)
        with open(sample_matrix, "rb") as datafile:
            row_ind, col_ind, dat = pickle.load(datafile)

        matrsize = max(row_ind) + 1

        return matrsize
