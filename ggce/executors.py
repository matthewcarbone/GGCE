#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import copy
from itertools import product
import numpy as np
import sys
from pathlib import Path
import time
import pickle
import yaml
import uuid

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ggce.engine.structures import SystemParams
from ggce.engine import system
from ggce.utils import utils

PRINT_EVERY_PERCENT = 5.0


def dryrun_random_result():
    """Returns a random value for G, 0.0 for the elapsed time, and 0 for
    the maximum matrix size."""

    G = np.abs(np.random.random()) + np.abs(np.random.random()) * 1j
    return (G, 0.0, 0)


class BaseExecutor:

    def __init__(self, rank_tool, pkg, cfg, solver, dry_run, nbuff):
        self.rank_tool = rank_tool
        self.logger = rank_tool.logger
        self.SIZE = rank_tool.SIZE
        self.RANK = rank_tool.RANK
        self.dry_run = dry_run
        self.solver = solver
        self.nbuff = nbuff
        self.pkg = pkg
        self.cfg = cfg
        self.trg = self.pkg / Path("results") / Path(self.cfg.stem)
        self.state_dir = self.trg / "STATE"
        self.done_file = self.trg / "DONE"

        # Initialize the input parameters
        self.inp = SystemParams(yaml.safe_load(open(self.cfg)))
        self.inp.prime()

    def prime_system(self):
        """Prepares the system object by using the stored input parameters."""

        t0 = time.time()
        if self.RANK == 0:
            nm = f"{self.inp.M}/{self.inp.N}"
            mod = self.inp.models
            self.logger.info(f"Priming with M/N = {nm}, model={mod}")
        sy = system.System(self.inp)
        T = sy.initialize_generalized_equations()
        L = sy.initialize_equations()
        sy.generate_unique_terms()
        sy.prime_solver()
        dt = (time.time() - t0) / 60.0
        return (sy, dt, T, L)


class Executor(BaseExecutor):
    """Executor class for running a standard k-point/w-point calculation. Runs
    in Theta(Nk * Nw) computational complexity.

    Parameters
    ----------
    mpi_info : RankTools
        Helper class containing information about the communicator and contains
        the logger on this rank.
    package_path : str
    config_path : str
        Location of the particular config file to load.
    solver : int
        The solver type. 0 for contiued fraction, 1 for direct sparse.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_remaining_jobs(self):
        """Loads in all indexes of the already completed jobs.
        Note set behavior: set([1, 2, 3]) - set([1, 4, 5, 6]) = {2, 3}
        """

        checkpoints = utils.listdir_fullpath(self.state_dir)
        completed_jobs = []
        for f in checkpoints:
            loaded_jobs = pickle.load(open(f, 'rb'))

            # The first two points are the k and w points
            completed_jobs.extend([tuple(ll[:2]) for ll in loaded_jobs])

        self.jobs = list(set(self.jobs) - set(completed_jobs))

    def prep_jobs(self, k_grid, w_grid):
        """Prepares the jobs to run by assigning each MPI process a chunk of
        the total job list."""

        jobs = list(product(k_grid, w_grid))
        jobs = self.rank_tool.chunk_jobs(jobs)
        self.jobs = jobs[self.RANK]

    def finalize(self, to_concat_on_rank):
        """Cleans up the state files by concatenating them. Each rank will
        read in a random sample of the data saved to STATE, and concatenate
        them into numpy arrays. Those numpy arrays will then be passed to
        the 0th rank, concatenated one more time, and saved to disk."""

        final = []
        for f in to_concat_on_rank:
            final.extend(pickle.load(open(f, 'rb')))

        return np.array(final)

    def save_final(self, arr):
        """Saves the final res.npy file. Must be executed on rank 0. If it
        is called on another rank it will silently do nothing."""

        if self.RANK != 0:
            return

        res_file = self.trg / Path("res.npy")
        with open(res_file, 'wb') as f:
            np.save(f, arr)

    def cleanup(self, to_delete_on_rank):
        """Removes all STATE files and saves the donefile."""

        for f in to_delete_on_rank:
            Path(f).unlink()

        with open(self.done_file, 'a') as f:
            f.write(f"RANK {self.RANK:05} tagged\n")

    def calculate(self, k_grid, w_grid):
        """If there are any to run, executes the calculations. Returns the
        total elapsed time of the computations."""

        # First, check if everything is done on this config.
        if self.done_file.exists():
            self.logger.warning(f"DONE file exists {self.cfg}")
            return 0.0

        # 8 decimal precision
        self.prep_jobs(k_grid, w_grid)

        # Check if there are remaining jobs on this rank
        if len(self.jobs) == 0:
            self.logger.warning(f"No jobs to run {self.cfg}")
            return 0.0

        # Load in all jobs which have been completed and get the remaining
        # jobs to run on this rank by comparing the sets. This will modify
        # the jobs attribute.
        self.find_remaining_jobs()

        # Construct the size of the buffer. To avoid lots of read/write
        # operations (especially when checkpointing), jobs will be buffered
        # so every N_buff jobs information will be pickled to the STATE
        # directory.
        if self.nbuff > 0:
            nbuff = self.nbuff
        else:
            nbuff = int(max(len(self.jobs) // 100, 1))

        buffer = utils.Buffer(nbuff, self.state_dir)

        # Prepare the system object. We disable the system logger unless on
        # rank 0 so as to reduce bloat to the output stream.
        sy = None
        if not self.dry_run:
            if self.RANK == 0:
                (sy, dt, T, L) = self.prime_system()
            else:
                with utils.DisableLogger():
                    (sy, dt, T, L) = self.prime_system()
        elif self.RANK == 0:
            self.logger.warning(
                "Running in dry run mode: G is randomly generated"
            )

        # Get the total number of jobs
        L = len(self.jobs)
        print_every = int(max(L * PRINT_EVERY_PERCENT / 100.0, 1))
        if self.RANK == 0:
            self.logger.info(f"Printing every {print_every} jobs")

        # Main calculation loop. Only jobs that need to be run are included
        # in the jobs attribute.
        all_results = []
        overall_config_time = time.time()
        for cc, (_k, _w) in enumerate(self.jobs):

            # Solve the system
            if not self.dry_run:
                with utils.DisableLogger():
                    G, meta = sy.solve(_k * np.pi, _w, self.solver)
                A = -G.imag / np.pi
                tcomp = meta['time'][-1] / 60.0
                largest_mat_dim = meta['inv'][0]
                self.logger.debug(
                    f"Solved A({_k:.02f}pi, {_w:.02f}) "
                    f"= {A:.02f} in {tcomp:.02f} m"
                )

                if A < 0.0:
                    self.logger.error(f"Negative spectral weight: {A:.02e}")
                    sys.stdout.flush()

                if (cc % print_every == 0 or cc == 0) and self.RANK == 0:
                    self.logger.info(f"{cc:05}/{L:05} done in {tcomp:.02f} m")
                    sys.stdout.flush()

            else:
                G, tcomp, largest_mat_dim = Executor.dryrun_random_result()

            val = [_k, _w, G.real, G.imag, tcomp, largest_mat_dim]
            all_results.append(val)

            if self.RANK == 0 and cc == 0:
                est_size = largest_mat_dim**2 * 16.0 / 1e9
                self.logger.info(f"Largest matrix size: {est_size:.02f} GB")

            # Buffer will automatically flush
            buffer(val)

        # Flush the buffer manually at the end if necessary
        buffer.flush()
        sys.stdout.flush()

        return time.time() - overall_config_time


class LowestBandExecutor(BaseExecutor):
    """Executor class for running a lowest-energy band calculation.

    Parameters
    ----------
    mpi_info : RankTools
        Helper class containing information about the communicator and contains
        the logger on this rank.
    package_path : str
    config_path : str
        Location of the particular config file to load.
    solver : int
        The solver type. 0 for contiued fraction, 1 for direct sparse.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gs_peak(self, _result):
        """Provides an estimate for the peak location, peak maximum and
        a guess for the next w-grid."""

        result = copy.deepcopy(_result)

        # Length of the w-grid
        N = result.shape[0]

        # Sort by omega
        result = result[result[:, 1].argsort()]
        tmp_w = result[:, 1]
        tmp_A = -result[:, 3] / np.pi

        try:
            peak = find_peaks(tmp_A)[0][0]
        except IndexError:
            self.logger.warning("Peak finder failed")
            peak = np.argmax(tmp_A)

        # Fit the ground state peak to a Lorentzian
        popt, _ = curve_fit(
            utils.lorentzian,
            tmp_w[peak-5:peak+5],
            tmp_A[peak-5:peak+5],
            p0=[tmp_w[peak], tmp_A[peak], self.inp.eta]
        )

        # The new w_grid should be around the computed maximum
        mu = popt[0]
        mag = popt[1]
        w_grid = np.linspace(
            mu - 2.0 * self.inp.eta, mu + 2.0 * self.inp.eta, N
        )

        self.logger.info(f"Peak found at (w, A) = ({mu:.02f}, {mag:.02f})")

        return w_grid

    def recalculate_k_grid(self, k_grid, w_grid_original):
        """The LowestBandExecutor distributes jobs only in the w grid. The
        k-grid is in a sense serial now, since the location of the w grid for
        k1 will depend on k0."""

        jobs = list(k_grid)

        checkpoints = utils.listdir_fullpath(self.state_dir)

        if len(checkpoints) == 0:
            return k_grid, w_grid_original

        completed_results = dict()
        for f in checkpoints:
            loaded = pickle.load(open(f, 'rb'))
            completed_results = {**completed_results, **loaded}

        completed_jobs = list(completed_results.keys())
        new_k_grid = list(set(jobs) - set(completed_jobs))
        new_k_grid.sort()

        completed_jobs.sort()
        last_k = completed_jobs[-1]
        new_w_grid = self.gs_peak(completed_results[last_k])

        return new_k_grid, new_w_grid

    def save(self, res):
        """Only to be used by rank 0, this saves the result for some k
        point.

        Parameters
        ----------
        res : np.array
            A numpy array where rows are different w-points and columns
            are different results and metadata.
        """

        if self.RANK != 0:
            return

        path = Path(self.state_dir) / Path(f"{uuid.uuid4().hex}.pkl")

        # The first column is the k-point
        k = res[:, 0]
        assert len(np.unique(k)) == 1
        k = k[0]

        pickle.dump({k: res}, open(path, 'wb'), protocol=4)

    def finalize(self, to_concat_on_rank):
        """Cleans up the state files by concatenating them. Each rank will
        read in a random sample of the data saved to STATE, and concatenate
        them into numpy arrays. Those numpy arrays will then be passed to
        the 0th rank, concatenated one more time, and saved to disk."""

        final = []
        for f in to_concat_on_rank:
            loaded = pickle.load(open(f, 'rb'))
            final.append(list(loaded.values())[0])

        return np.concatenate(final, axis=0)

    def save_final(self, arr):
        """Saves the final res.npy file. Must be executed on rank 0. If it
        is called on another rank it will silently do nothing."""

        if self.RANK != 0:
            return

        res_file = self.trg / Path("res.npy")
        with open(res_file, 'wb') as f:
            np.save(f, arr)

    def cleanup(self, to_delete_on_rank):
        """Removes all STATE files and saves the donefile."""

        for f in to_delete_on_rank:
            Path(f).unlink()

        with open(self.done_file, 'a') as f:
            f.write(f"RANK {self.RANK:05} tagged\n")

    def prep_jobs(self, w_grid):
        """Prepares the jobs to run by assigning each MPI process a chunk of
        the total job list."""

        jobs = list(w_grid)
        jobs = self.rank_tool.chunk_jobs(jobs)
        self.jobs = jobs[self.RANK]

    def prime_calculate(self):

        # Prepare the system object. We disable the system logger unless on
        # rank 0 so as to reduce bloat to the output stream.
        self.sy = None
        if not self.dry_run:
            if self.RANK == 0:
                (self.sy, _, _, _) = self.prime_system()
            else:
                with utils.DisableLogger():
                    (self.sy, _, _, _) = self.prime_system()
        elif self.RANK == 0:
            self.logger.warning(
                "Running in dry run mode: G is randomly generated"
            )

    def calculate(self, _k, w_grid):
        """If there are any to run, executes the calculations. Returns the
        total elapsed time of the computations."""

        # First, check if everything is done on this config.
        if self.done_file.exists():
            self.logger.warning(f"DONE file exists {self.cfg}")
            return None, 0.0

        self.prep_jobs(w_grid)

        # Check if there are remaining jobs on this rank
        if len(self.jobs) == 0:
            self.logger.warning(f"No jobs to run {self.cfg}")
            return None, 0.0

        # Get the total number of jobs
        L = len(self.jobs)
        print_every = int(max(L * PRINT_EVERY_PERCENT / 100.0, 1))
        if self.RANK == 0:
            self.logger.info(f"Printing every {print_every} jobs")

        # Main calculation loop. Only jobs that need to be run are included
        # in the jobs attribute.
        all_results = []
        overall_config_time = time.time()
        for cc, _w in enumerate(self.jobs):

            # Solve the system
            if not self.dry_run:
                with utils.DisableLogger():
                    G, meta = self.sy.solve(_k * np.pi, _w, self.solver)
                A = -G.imag / np.pi
                tcomp = meta['time'][-1] / 60.0
                largest_mat_dim = meta['inv'][0]
                self.logger.debug(
                    f"Solved A({_k:.02f}pi, {_w:.02f}) "
                    f"= {A:.02f} in {tcomp:.02f} m"
                )

                if A < 0.0:
                    self.logger.error(f"Negative spectral weight: {A:.02e}")
                    sys.stdout.flush()

                if (cc % print_every == 0 or cc == 0) and self.RANK == 0:
                    self.logger.info(f"{cc:05}/{L:05} done in {tcomp:.02f} m")
                    sys.stdout.flush()

            else:
                G, tcomp, largest_mat_dim = Executor.dryrun_random_result()

            val = [_k, _w, G.real, G.imag, tcomp, largest_mat_dim]
            all_results.append(val)

            if self.RANK == 0 and cc == 0:
                est_size = largest_mat_dim**2 * 16.0 / 1e9
                self.logger.info(f"Largest matrix size: {est_size:.02f} GB")

        sys.stdout.flush()

        return np.array(all_results), time.time() - overall_config_time
