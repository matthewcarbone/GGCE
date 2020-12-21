#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

"""
Run with, e.g.,
---------------
mpiexec -np 4 python3 ._submit.py /Users/mc/Data/scratch/GGCE/000_TEST2 1

"""

from itertools import product
import numpy as np
import os
import pickle
import sys
import time
import uuid
import yaml

from mpi4py import MPI

from ggce.engine.structures import InputParameters
from ggce.engine import system
from ggce.utils.logger import default_logger as _dlog
from ggce.utils import utils


PRINT_EVERY_PERCENT = 10.0


class LoggerOnRank:

    def __init__(self, rank, logger, debug_flag=False):
        self.rank = rank
        self.logger = logger
        self.debug_flag = debug_flag

    def debug(self, msg):
        if self.debug_flag:
            self.logger.debug(f"({self.rank:05}) {msg}")

    def info(self, msg):
        self.logger.info(f"({self.rank:05}) {msg}")

    def warning(self, msg):
        self.logger.warning(f"({self.rank:05}) {msg}")

    def error(self, msg):
        self.logger.error(f"({self.rank:05}) {msg}")

    def critical(self, msg):
        self.logger.critical(f"({self.rank:05}) {msg}")


class RankTools:
    """A helper class containing information about the current MPI communicator
    as well as the rank, and logger."""

    def __init__(self, communicator, logger, debug):
        self.SIZE = communicator.size
        self.RANK = communicator.rank
        self.logger = \
            LoggerOnRank(rank=self.RANK, logger=logger, debug_flag=bool(debug))

    def chunk_jobs(self, jobs):
        """Returns self.SIZE chunks, each of which is a list which is a
        reasonably equally distributed representation of jobs."""

        return [jobs[ii::self.SIZE] for ii in range(self.SIZE)]


class Buffer:

    def __init__(self, nbuff, target_directory):
        self.nbuff = nbuff
        self.counter = 0
        self.queue = []
        self.target_directory = target_directory

    def flush(self):
        if self.counter > 0:
            f = f"{uuid.uuid4().hex}.pkl"
            path = os.path.join(self.target_directory, f)
            pickle.dump(self.queue, open(path, 'wb'), protocol=4)
            self.counter = 0
            self.queue = []

    def __call__(self, val):
        self.queue.append(val)
        self.counter += 1
        if self.counter >= self.nbuff:
            self.flush()


class Executor:
    """

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

    @staticmethod
    def dryrun_random_result():
        """Returns a random value for G, 0.0 for the elapsed time, and 0 for
        the maximum matrix size."""

        G = np.abs(np.random.random()) + np.abs(np.random.random()) * 1j
        return (G, 0.0, 0)

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
        return jobs[self.RANK]

    def __init__(self, rank_tool, package_path, config_path, solver, dry_run):
        self.rank_tool = rank_tool
        self.logger = rank_tool.logger
        self.SIZE = rank_tool.SIZE
        self.RANK = rank_tool.RANK
        self.dry_run = dry_run
        self.solver = solver
        base = os.path.splitext(os.path.basename(config_path))[0]
        self.config_path = config_path
        self.target_dir = os.path.join(package_path, "results", base)
        self.state_dir = os.path.join(self.target_dir, "STATE")
        self.done_file = os.path.join(self.target_dir, "DONE")

    def check_done_file(self):
        """Checks the calculation directory for a file DONE. If it exists,
        it indicates that all computations have been saved to res.txt for this
        config."""

        if os.path.isfile(self.done_file):
            return True
        return False

    def prime_system(self):
        """Prepares the system object by using the stored input parameters."""

        t0 = time.time()
        nm = f"{self.inp.model_params.M}/{self.inp.model_params.N}"
        mod = self.inp.model_params.model
        if self.RANK == 0:
            self.logger.info(f"Priming with M/N = {nm}, model={mod}")
        sy = system.System(self.inp)
        T = sy.initialize_generalized_equations()
        L = sy.initialize_equations()
        sy.generate_unique_terms()
        sy.prime_solver()
        dt = (time.time() - t0) / 60.0
        return (sy, dt, T, L)

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

        res_file = os.path.join(self.target_dir, "res.npy")
        with open(res_file, 'wb') as f:
            np.save(f, arr)

    def cleanup(self, to_delete_on_rank):
        """Removes all STATE files and saves the donefile."""

        for f in to_delete_on_rank:
            os.remove(f)

        with open(self.done_file, 'a') as f:
            f.write(f"RANK {self.RANK:05} tagged\n")

    def calculate(self):
        """If there are any to run, executes the calculations. Returns the
        total elapsed time of the computations."""

        # First, check if everything is done on this config.
        if self.check_done_file():
            self.logger.warning(f"DONE file exists {self.config_path}")
            return -1

        # Initialize the input parameters
        self.inp = InputParameters(yaml.safe_load(open(self.config_path)))
        self.inp.prime()

        # 8 decimal precision
        w_grid = np.round(self.inp.get_w_grid(), 8)
        k_grid = np.round(self.inp.get_k_grid(), 8)  # In units of pi!
        self.jobs = self.prep_jobs(k_grid, w_grid)

        # Check if there are remaining jobs on this rank
        if len(self.jobs) == 0:
            self.logger.warning(f"No jobs to run {self.config_path}")
            return

        # Load in all jobs which have been completed and get the remaining
        # jobs to run on this rank by comparing the sets. This will modify
        # the jobs attribute.
        self.find_remaining_jobs()

        # Construct the size of the buffer. To avoid lots of read/write
        # operations (especially when checkpointing), jobs will be buffered
        # so every N_buff jobs information will be pickled to the STATE
        # directory.
        nbuff = int(max(len(self.jobs) // 100, 1))
        if self.RANK == 0:
            self.logger.info(f"Buffer will flush every {nbuff} results")
        buffer = Buffer(nbuff, self.state_dir)

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
        overall_config_time = time.time()
        for cc, (_k, _w) in enumerate(self.jobs):

            # Solve the system
            if not dry_run:
                with utils.DisableLogger():
                    G, meta = sy.solve(_k * np.pi, _w, self.solver)
                A = -G.imag / np.pi
                computation_time = meta['time'][-1] / 60.0
                largest_mat_dim = meta['inv'][0]
                self.logger.debug(
                    f"Solved A({_k:.02f}pi, {_w:.02f}) "
                    f"= {A:.02f} in {computation_time:.02f} m"
                )

                if A < 0.0:
                    self.logger.error(f"Negative spectral weight: {A:.02e}")
                    sys.stdout.flush()

                if (cc % print_every == 0 or cc == 0) and self.RANK == 0:
                    self.logger.info(
                        f"{cc:05}/{L:05} done in {computation_time:.02f} m"
                    )
                    sys.stdout.flush()

            else:
                G, computation_time, largest_mat_dim = \
                    Executor.dryrun_random_result()

            val = [_k, _w, G.real, G.imag, computation_time, largest_mat_dim]

            # Buffer will automatically flush
            buffer(val)

        # Flush the buffer manually at the end if necessary
        buffer.flush()
        sys.stdout.flush()

        return time.time() - overall_config_time


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD  # Default MPI communicator

    # The first argument passed is the base path for the calculation.
    package_path = str(sys.argv[1])

    # The second argument is if to run in debug mode or not
    debug = int(sys.argv[2])

    # The third argument is whether to run in dry run mode or now
    dry_run = int(sys.argv[3])

    # Type of solver
    solver = int(sys.argv[4])

    # MPI info includes the logger on that rank
    mpi_info = RankTools(COMM, _dlog, debug)

    if mpi_info.RANK == 0:
        # Load in the data, which will be used to process into the jobs. The
        # master list contains the list of all omega -> config key pairs (which
        # correspond to the config mapping). The config_mapping maps a number
        # to an actual python dictionary containing the configuration for that
        # trial. The N_M_eta_permutations is a list of the (M, N, eta) to run,
        # and the package_cache_path is the base cache path.
        COMM_timer = time.time()
        _all_configs_paths = utils.listdir_fullpath(
            os.path.join(package_path, "configs")
        )

        # Remove any config with a donefile
        # base = os.path.splitext(os.path.basename(config_path))[0]
        # self.config_path = config_path
        # self.target_dir = os.path.join(package_path, "results", base)
        # self.state_dir = os.path.join(self.target_dir, "STATE")
        # self.done_file = os.path.join(self.target_dir, "DONE")
        all_configs_paths = []
        for config in _all_configs_paths:
            base = os.path.splitext(os.path.basename(config))[0]
            target_dir = os.path.join(package_path, "results", base)
            done_file = os.path.join(target_dir, "DONE")
            if not os.path.isfile(done_file):
                all_configs_paths.append(config)

        all_configs_paths.sort()
        mpi_info.logger.info(f"Confirming COMM world size: {mpi_info.SIZE}")
        mpi_info.logger.info(f"Running {len(all_configs_paths)} config files")
        mpi_info.logger.info(f"Will use solver type {solver}")
        mpi_info.logger.info(f"Dryrun is {dry_run}; debug is {debug}")
    else:
        COMM_timer = None
        all_configs_paths = None

    rank_timer = time.time()

    # Iterate over the config files
    all_configs_paths = COMM.bcast(all_configs_paths, root=0)
    for config_path in all_configs_paths:

        # Startup the Executor, which is a helper class for running the
        # calculation using an MPI implementation
        executor = Executor(
            mpi_info, package_path, config_path, solver, dry_run
        )

        # ---------------------------------------------------------------------
        # CALCULATE -----------------------------------------------------------

        # Run the calculation on this rank
        elapsed = executor.calculate()

        # Collect the runtimes for each of these processes. This also serves
        # as a barrier.
        elapsed = COMM.gather(elapsed, root=0)

        # Print some useful information about how fast the overall process
        # was and how imbalanced the loads were
        if mpi_info.RANK == 0:
            avg = np.mean(elapsed) / 60.0
            sd = np.std(elapsed) / 60.0
            mpi_info.logger.info(
                f"CALCULATE done in {avg:.02f} +/- {sd:.02f} m"
            )
        COMM.Barrier()

        # ---------------------------------------------------------------------
        # FINALIZE ------------------------------------------------------------

        # Let the 0th rank list all files in the current config directory and
        # scatter them to the respective ranks.
        if mpi_info.RANK == 0:
            tmp_t = time.time()
            state_files = utils.listdir_fullpath(executor.state_dir)
            state_files = mpi_info.chunk_jobs(state_files)
        else:
            state_files = None
        state_files = COMM.scatter(state_files, root=0)

        # Begin the concatenation process of collecting all of the STATE files
        res = executor.finalize(state_files)
        res = COMM.gather(res, root=0)

        if mpi_info.RANK == 0:
            tmp_t = (time.time() - tmp_t) / 60.0
            mpi_info.logger.info(f"FINALIZE done in {tmp_t:.02f} m")

        # Concatenate the results on rank 0 and save to disk
        if mpi_info.RANK == 0:
            res = np.concatenate(res, axis=0)
            executor.save_final(res)
        COMM.Barrier()

        # ---------------------------------------------------------------------
        # CLEANUP -------------------------------------------------------------

        if mpi_info.RANK == 0:
            tmp_t = time.time()

        executor.cleanup(state_files)
        COMM.Barrier()

        # Last step is to delete STATE
        if mpi_info.RANK == 0:
            os.rmdir(executor.state_dir)

        if mpi_info.RANK == 0:
            tmp_t = (time.time() - tmp_t) / 60.0
            mpi_info.logger.info(f"CLEANUP done in {tmp_t:.02f} m")
        COMM.Barrier()

    COMM.Barrier()
    if mpi_info.RANK == 0:
        time.sleep(1)
        dt = (time.time() - COMM_timer) / 3600.0
        mpi_info.logger.info(f"ALL done in {dt:.02f} h")
