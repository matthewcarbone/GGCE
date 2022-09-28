from pathlib import Path
import numpy as np
import time, pickle

from petsc4py import PETSc

from ggce.executors.petsc4py.base import MassiveSolver
from ggce.logger import logger, disable_logger

BYTES_TO_GB = 1073741274

class MassiveSolverMUMPS(MassiveSolver):
    """A class to connect to PETSc powerful parallel sparse solver tools, to
    calculate G(k,w) in parallel, using a one-shot sparse solver MUMPS.
    This inherits the matrix construction strategies of the BaseExecutorPETSC
    base class.

    This is done because e.g. convergence and memory checks are often specific
    to the particular solver used -- and so is the KSP context (i.e. solver)
    setup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._basis is None:
            self._basis = self._system.get_basis(full_basis=True)

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
                logger.debug(
                    "According to MUMPS diagnostics, call to MUMPS was "
                    f"successful. The calculation took {elapsed:.2f} sec."
                )
            elif self.mumps_conv_ind < 0:
                logger.error(
                    "A MUMPS error occured with MUMPS error code "
                    f"{self.mumps_conv_ind} See the MUMPS User Guide, Sec. 8, "
                    "for error  diagnostics. The calculation took "
                    f"{elapsed:.2f} sec."
                )
            elif self.mumps_conv_ind > 0:
                logger.warning(
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
        logger.debug(
            f"Current rank MUMPS memory usage is {self.rank_mem_used:.02f} GB"
        )

        # set up memory usage tracking, report to the logger on head node only
        # total memory across all processes
        self.total_mem_used = factored_mat.getMumpsInfog(31) * 1e6 \
            / BYTES_TO_GB
        if self.mpi_rank == 0:
            logger.debug(
                f"Total MUMPS memory usage is {self.total_mem_used:.02f} GB"
            )

    def _pre_solve(self, k, w, eta):
        result = None
        path = None
        if self._results_directory is not None:
            ckpt_path = f"{self._k_omega_eta_to_str(k, w, eta)}.pckl"
            path = self._results_directory / Path(ckpt_path)
            if path.exists():
                result = np.array(pickle.load(open(path, "rb")))
        return result, path

    def _post_solve(self, G, k, w, path):
        if -G.imag / np.pi < 0.0:
            logger.error(f"A(k,w) < 0 at k, w = ({k:.02f}, {w:.02f}")
        if self._results_directory is not None:
            pickle.dump(G, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # @profile
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
            using the PETSc KSP context and MUMPS memory and exit codes.
        """

        # first check if you already calculated this
        result, path = self._pre_solve(k, w, eta)
        if result is not None:
            return result

        if self.basis_dir is None:
            self._scaffold(k, w, eta)
        else:
            self._scaffold_from_disk(k, w, eta, basis_dir = self.basis_dir)

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
        logger.debug("KSP and PC contexts initialized", elapsed=dt)

        # Call the solve method
        t0 = time.time()
        ksp.solve(self._vector_b, self._vector_x)
        dt = time.time() - t0

        # assemble the solution vector
        self._vector_x.assemblyBegin()
        self._vector_x.assemblyEnd()

        self._mpi_comm_brigadier.barrier()

        # call manual residual check, as well as check MUMPS INFO
        # if the MUMPS solver call was successful
        # to get access to MUMPS error codes, get the factored matrix from PC
        factored_mat = pc.getFactorMatrix()
        self.check_conv_manual(pc, rtol=rtol)
        self.check_conv(factored_mat, rtol=rtol, elapsed=dt)

        # now check memory usage
        self.check_mem_use(factored_mat)

        self._mpi_comm_brigadier.barrier()

        # for memory management, destroy the KSP context manually
        ksp.destroy()

        # The last rank has the end of the solution vector, which contains G
        # G is the last entry aka "the last equation" of the matrix
        # use a gather operation, called by all ranks, to construct the full
        # vector (currently not used but will be later)
        G_vec = self._mpi_comm_brigadier.gather(self._vector_x.getArray(), root=0)

        # since we grabbed the Green's func value, destroy the data structs
        # self._vector_x.destroy()
        # self._vector_b.destroy()
        # self._mat_X.destroy()

        # Now select only the final value from the array
        if self.brigade_rank == 0:
            G_val = G_vec[self.brigade_size-1][-1]
        else:
            G_val = None

        # and bcast to all processes in your brigade
        G_val = self._mpi_comm_brigadier.bcast(G_val, root=0)

        # only checkpoint if you are the sergeant
        if self.brigade_rank == 0:
            self._post_solve(G_val, k, w, path)

        return np.array(G_val), {
            'time': [dt],
            'mumps_exit_code': [self.mumps_conv_ind],
            'mumps_mem_tot': [self.total_mem_used],
            'manual_tolerance_excess': [self.tol_excess]
        }
