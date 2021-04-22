#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


from itertools import product
import numpy as np
import sys
from pathlib import Path
import time
import pickle
import yaml

from ggce.engine.structures import SystemParams
from ggce.engine import system
from ggce.utils import utils

PRINT_EVERY_PERCENT = 5.0


def dryrun_random_result():
    """Returns a random value for G, 0.0 for the elapsed time, and 0 for
    the maximum matrix size."""

    G = np.abs(np.random.random()) + np.abs(np.random.random()) * 1j
    return (G, 0.0, 0)


def finalize_lowest_band_executor(state_dir, trg, done_file, rank=0):
    """Finishes the LowestBandExecutor calculation by saving the resulting
    pickle file.

    Parameters
    ----------
    state_dir : str
        The location of the STATE directory.
    trg : str
        The location of the results directory
    done_file : str
        The location of the DONE file.
    rank : int, optional
        MPI rank (the default is 0).

    Returns
    -------
    list
        The results (all spectra, last data points, peak locations and qp
        weights).
    """

    checkpoints = utils.listdir_fullpath(state_dir)
    checkpoints.sort()
    all_spectra = []
    all_last_datapoints = []
    all_peak_locations = []
    all_qp_weights = []
    for f in checkpoints:
        loaded = pickle.load(open(f, 'rb'))
        k = list(loaded.keys())[0]

        try:
            all_spectra.extend(loaded[k][0])
            all_last_datapoints.append(loaded[k][1][0])
            all_peak_locations.append(loaded[k][1][1])
            all_qp_weights.append(loaded[k][1][2])

        # TypeError: 'NoneType' object is not subscriptable
        except TypeError:
            break

    all_spectra = np.array(all_spectra)

    res_file = trg / Path("res_gs.pkl")
    r = [
        all_spectra, all_last_datapoints, all_peak_locations,
        all_qp_weights
    ]
    pickle.dump(r, open(res_file, 'wb'), protocol=4)

    for f in checkpoints:
        Path(f).unlink()

    with open(done_file, 'a') as f:
        f.write(f"RANK {rank:05} tagged\n")

    return r


class BaseExecutor:

    def __init__(self, rank_tool, pkg, cfg, solver, dry_run, nbuff):
        self.rank_tool = rank_tool
        self.logger = rank_tool.logger
        self.SIZE = rank_tool.rank
        self.RANK = rank_tool.rank
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

            # Log all of the information about the run!
            # nm = f"{self.inp.M}/{self.inp.N}"
            mod = ", ".join(self.inp.models_vis)
            self.logger.info(f"MODELS: [{mod}]")
            _Omega = ", ".join([f"{o:.02f}" for o in self.inp.Omega])
            self.logger.info(f"O = [{_Omega}]")
            _lambdas = ", ".join([f"{o:.02f}" for o in self.inp.lambdas])
            self.logger.info(f"L = [{_lambdas}]")
            self.logger.info(f"T = {self.inp.temperature:.02f}")
            self.logger.info(f"M = {self.inp.M}")
            self.logger.info(f"N = {self.inp.N}")

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
        The solver type. 0 for continued fraction, 1 for direct sparse.
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
    """Executor class for running a lowest-energy band calculation. This
    calculation runs in serial in w and k.

    Parameters
    ----------
    mpi_info : RankTools
        Helper class containing information about the communicator and contains
        the logger on this rank.
    package_path : str
    config_path : str
        Location of the particular config file to load.
    solver : int
        The solver type. 0 for continued fraction, 1 for direct sparse.
    """

    def __init__(
        self, rank_tool, pkg, cfg, solver, dry_run, eta_div, eta_step_div,
        next_k_offset_factor
    ):
        super().__init__(rank_tool, pkg, cfg, solver, dry_run, None)

        # Iteration for w grid
        assert eta_step_div > 0.0
        self.eta_step = self.inp.eta / eta_step_div
        self.eta_prime = self.inp.eta / eta_div
        self.next_k_offset_factor = next_k_offset_factor

    def finalize(self):
        """Loads in everything and re-saves as res_gs.pkl"""

        finalize_lowest_band_executor(
            self.state_dir, self.trg, self.done_file, rank=self.RANK
        )

    def recalculate_grids(self, w0):
        """The LowestBandExecutor distributes jobs only in the w grid. The
        k-grid is in a sense serial now, since the location of the w grid for
        k1 will depend on k0."""

        checkpoints = utils.listdir_fullpath(self.state_dir)

        # All jobs must be run
        if len(checkpoints) == 0:
            return 0, w0, []

        completed_results = dict()
        for f in checkpoints:
            loaded = pickle.load(open(f, 'rb'))
            completed_results = {**completed_results, **loaded}

        # Iterate through completed results.
        # The k_index simply iterates the index on the pre-defined k-grid
        # Results is a list of lists, where each entry is like [k, w, G, ...]
        # xc and gs are extra information, particularly the extra calculation
        # corresponding to the different eta used to get the peak location,
        # and quasiparticle weight.
        for k_index, [results, final] in completed_results.items():
            if final is None:
                resume_w_at = results[-1][1]  # This is the last omega point
                return k_index, resume_w_at + self.eta_step, results

        completed_jobs = list(completed_results.keys())
        completed_jobs.sort()
        last_k = completed_jobs[-1]
        last_w = completed_results[last_k][0][-1][1]
        resume_w = last_w - self.next_k_offset_factor * self.inp.eta
        return last_k + 1, resume_w, []

    def save(self, k_index, results, final=None):
        """Saves the results (or result) for some k-point.

        Parameters
        ----------
        k_index : int
            The index of the kpoint.
        res : np.array
            A numpy array where rows are different w-points and columns
            are different results and metadata. Contains all of the results
            so far.
        final
            Either None or contains:
            extra_calculation : np.array
                When the maximum is found, an extra calculation is recorded
                with a slightly smaller eta.
            gs : float
                The location of the ground state peak.
            qp_weight : float
                The quasi particle weight.
        """

        if self.RANK != 0:
            return

        # path = Path(self.state_dir) / Path(f"{uuid.uuid4().hex}.pkl")

        path = Path(self.state_dir) / Path(f"k={k_index:08}.pkl")

        pickle.dump({k_index: [results, final]}, open(path, 'wb'), protocol=4)

    def prime_calculate(self):

        # Prepare the system object. We disable the system logger unless on
        # rank 0 so as to reduce bloat to the output stream.
        if not self.dry_run:
            (self.sy, _, _, _) = self.prime_system()
        elif self.RANK == 0:
            self.logger.warning(
                "Running in dry run mode: G is randomly generated"
            )

    def peak_location_and_weight(self, w, A, Aprime):
        """Assumes that the polaron peak is a Lorentzian has the same weight
        no matter the eta. With these assumptions, we can determine the
        location and weight exactly using two points, each from a different
        eta calculation."""

        numerator1 = np.sqrt(self.inp.eta * self.eta_prime)
        numerator2 = (A * self.inp.eta - Aprime * self.eta_prime)
        den1 = Aprime * self.inp.eta - A * self.eta_prime
        den2 = A * self.inp.eta - Aprime * self.eta_prime
        loc = w - np.abs(numerator1 * numerator2 / np.sqrt(den1 * den2))
        area = np.pi * A * ((w - loc)**2 + self.inp.eta**2) / self.inp.eta
        return loc, area

    def calculate(self, _k, _w, use_eta_prime=False):
        """If there are any to run, executes the calculations. Returns the
        total elapsed time of the computations. If use_eta_prime=True, will run
        using self.sy_prime, which is special to the ground state computation.
        This runs using the second eta value specified by eta_div in the
        input file, and is ultimately used to compute exactly the peak position
        and weights, assuming the ground state is a perfect Lorentzian.

        Returns
        -------
        list, float
            The single result for the specified k and w point, in addition to
            the computation time.
        """

        # Main calculation loop. Only jobs that need to be run are included
        # in the jobs attribute.
        overall_config_time = time.time()

        # Solve the system
        if not self.dry_run:
            with utils.DisableLogger():
                if not use_eta_prime:
                    G, meta = self.sy.solve(_k * np.pi, _w, self.solver)
                else:
                    G, meta = self.sy.solve(
                        _k * np.pi, _w, self.solver, eta=self.eta_prime
                    )
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

            sys.stdout.flush()

        else:
            G, tcomp, largest_mat_dim = Executor.dryrun_random_result()

        val = [_k, _w, G.real, G.imag, tcomp, largest_mat_dim]
        return val, time.time() - overall_config_time
