from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ggce.logger import logger, disable_logger
from ggce.utils.utils import float_to_list, chunk_jobs
from ggce.engine.system import System
from ggce.utils.physics import G0_k_omega


BYTES_TO_MB = 1048576


class Solver(ABC):
    @property
    def system(self):
        return self._system

    @property
    def root(self):
        return self._root

    @property
    def basis(self):
        return self._basis

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

    def __init__(self, system=None, root=None, basis=None, mpi_comm=None):
        self._system = system
        self._root = root
        self._mpi_comm = mpi_comm
        self._basis = basis

        if self._system is None and self._root is None:
            logger.critical("Either system, root or both must be provided")

        if self._root is not None:
            # We allow checkpointing
            self._root = Path(self._root)
            self._results_directory = self._root / Path("results")
            self._results_directory.mkdir(exist_ok=True, parents=True)

        else:
            logger.warning("root not provided to Solver - Solver checkpointing disabled")
            self._results_directory = None

        if self._system is None:
            # Attempt to load the system from its checkpoint... the system
            # will now be initialized or an error will be thrown
            self._system = System.from_checkpoint(self._root)

        # Force checkpoint the system, which at this point must be initialized
        with disable_logger():
            self._system.checkpoint()

        if self._root is not None:
            logger.info(f"System checkpointed to '/{self._root}'")

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

    @staticmethod
    def _k_omega_eta_to_str(k, omega, eta):
        # Note this will have to be redone when k is a vector in 2 and 3D!
        return f"{k:.10f}_{omega:.10f}_{eta:.10f}"

    @abstractmethod
    def _pre_solve(self):
        ...

    @abstractmethod
    def _post_solve(self):
        ...

    @abstractmethod
    def solve(self, k, w, eta):
        """Takes, ``k, w, eta`` and returns the Green's function."""
        ...

    @abstractmethod
    def spectrum(self, k, w, eta, pbar=False):
        ...


class BasicSolver(Solver):
    def _pre_solve(self, k, w, eta):
        result = None
        path = None
        if self._results_directory is not None:
            ckpt_path = f"{self._k_omega_eta_to_str(k, w, eta)}.pkl"
            path = self._results_directory / Path(ckpt_path)
            if path.exists():
                result = np.array(pickle.load(open(path, "rb")))
        return result, path

    def _post_solve(self, G, k, w, path):
        if -G.imag / np.pi < 0.0:
            logger.error(f"A(k,w) < 0 at k, w = ({k:.02f}, {w:.02f}")
        if self._results_directory is not None:
            pickle.dump(G, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def spectrum(self, k, w, eta, pbar=False):
        """Solves for the spectrum in serial or in parallel, depending on
        whether MPI_COMM is provided to the Solver at instantiation.

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
        numpy.ndarray
            The resulting spectrum of shape ``nk`` by ``nw``.
        """

        k = float_to_list(k)
        w = float_to_list(w)

        # This orders the jobs such that when constructed into an array, the
        # k-points are the rows and the w-points are the columns after reshape
        jobs = [(_k, _w) for _k in k for _w in w]
        jobs_on_rank = self.get_jobs_on_this_rank(jobs)

        s = []
        for (_k, _w) in tqdm(jobs_on_rank, disable=not pbar):
            s.append(self.solve(_k, _w, eta))

        if self.mpi_comm is not None:
            all_results = self.mpi_comm.gather(s, root=0)

            # Only rank 0 returns the result
            if self.mpi_rank == 0:
                ## all_results is a list of arrays of different length
                ## need to parse it properly into an array
                all_results = [xx[ii] for xx in all_results \
                                                    for ii in range(len(xx))]
                s = np.array([xx for xx in all_results])
                return s.reshape(len(k), len(w))
            else:
                return None

        return np.array(s).reshape(len(k), len(w))


class SparseSolver(BasicSolver):
    """A sparse, serial solver for the Green's function. Useful for when the
    calculation being performed is quite cheap. Note that there are a variety
    of checkpointing features automatically executed when paths are provided.

    - When a ``System`` and path are provided, the system's root directory
      is overwritten and that object immediately checkpointed to the provided
      directory.

    - If only a ``System`` object is provided, no checkpointing will be
      performed.

    - If only a path is provided, the solver will attempt to instantiate the
      ``System`` object. Checkpointing will the proceed as normal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._basis is None:
            self._basis = self._system.get_basis(full_basis=True)

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
        est_mem_used = 24 * len(dat) / BYTES_TO_MB
        logger.debug(f"Estimated memory needed is {est_mem_used:.02f} MB")

        return row_ind, col_ind, dat

    def _scaffold(self, k, w, eta):
        """Prepare the X, v sparse representation of the matrix to solve.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation.

        Returns
        -------
        csr_matrix, csr_matrix
            Sparse representation of the matrix equation to solve, X and v.
        """

        row_ind, col_ind, dat = self._sparse_matrix_from_equations(k, w, eta)

        X = coo_matrix(
            (
                np.array(dat, dtype=np.complex64),
                (np.array(row_ind), np.array(col_ind)),
            )
        ).tocsr()

        size = (X.data.size + X.indptr.size + X.indices.size) / BYTES_TO_MB

        logger.debug(f"Memory usage of sparse X is {size:.01f} MB")

        # Initialize the corresponding sparse vector
        # {G}(0)
        row_ind = np.array([self._basis["{G}[0.0]"]])
        col_ind = np.array([0])
        v = coo_matrix(
            (
                np.array(
                    [self._system.equations[0][0].bias(k, w, eta)],
                    dtype=np.complex64,
                ),
                (row_ind, col_ind),
            )
        ).tocsr()

        return X, v

    def solve(self, k, w, eta):
        """Solve the sparse-represented system for some given point
        :math:`(k, \\omega, \\eta)`. Note that if the spectral function is
        negative results will not be checkpointed.

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
        numpy.ndarray
            The value of :math:`G(k, \\omega; \\eta)`.
        """

        result, path = self._pre_solve(k, w, eta)
        if result is not None:
            return result

        # Solution ------------------------------------------------------------
        X, v = self._scaffold(k, w, eta)
        res = spsolve(X, v)
        G = res[self._basis["{G}[0.0]"]]
        # Solution Done -------------------------------------------------------

        self._post_solve(G, k, w, path)

        return np.array(G)


class DenseSolver(BasicSolver):
    """A sparse, dense solver for the Green's function. Note that there are a
    variety of checkpointing features automatically executed when paths are
    provided. Uses the SciPy dense solver engine to solve for G(k, w) in
    serial. This method uses the continued fraction approach,

    .. math:: R_{n-1} = (1 - \\beta_{n-1}R_{n})^{-1} \\alpha_{n-1}

    with

    .. math:: R_n = \\alpha_n

    and

    .. math:: f_n = R_n f_{n-1}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._basis is None:
            self._basis = self._system.get_basis(full_basis=False)

    def _fill_matrix(self, k, w, n_phonons, shift, eta):

        n_phonons_shift = n_phonons + shift

        equations_n = self._system.equations[n_phonons]

        # Initialize a matrix to fill
        d1 = len(self._basis[n_phonons])
        d2 = len(self._basis[n_phonons + shift])
        A = np.zeros((d1, d2), dtype=np.complex64)

        # Fill the matrix of coefficients
        for ii, eq in enumerate(equations_n):
            index_term_id = eq.index_term.id()
            ii_basis = self._basis[n_phonons][index_term_id]
            for term in eq._terms_list:
                if term.config.total_phonons != n_phonons_shift:
                    continue
                t_id = term.id()
                jj_basis = self._basis[n_phonons_shift][t_id]
                A[ii_basis, jj_basis] += term.coefficient(k, w, eta)

        return A

    def _get_alpha(self, k, w, n_phonons, eta):
        return self._fill_matrix(k, w, n_phonons, -1, eta)

    def _get_beta(self, k, w, n_phonons, eta):
        return self._fill_matrix(k, w, n_phonons, 1, eta)

    def solve(self, k, w, eta):
        """Solve the dense-represented system.

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
        np.ndarray
            The value of G.
        """

        result, path = self._pre_solve(k, w, eta)
        if result is not None:
            return result

        # Solution ------------------------------------------------------------
        total_phonons = np.sum(self._system.model.phonon_number)
        for n_phonons in range(total_phonons, 0, -1):

            # Special case of the recursion where R_N = alpha_N.
            if n_phonons == total_phonons:
                R = self._get_alpha(k, w, n_phonons, eta)
                continue

            # Get the next loop's alpha and beta values
            beta = self._get_beta(k, w, n_phonons, eta)
            alpha = self._get_alpha(k, w, n_phonons, eta)

            # Compute the next R
            X = np.eye(beta.shape[0], R.shape[1]) - beta @ R
            R = linalg.solve(X, alpha)

        a = self._system.model.lattice_constant
        t = self._system.model.hopping
        G0 = G0_k_omega(k, w, a, eta, t)

        beta0 = self._get_beta(k, w, 0, eta)
        G = (G0 / (1.0 - beta0 @ R)).squeeze()
        G = np.array(G, dtype=np.complex64)
        # Solution Done -------------------------------------------------------

        self._post_solve(G, k, w, path)

        return G
