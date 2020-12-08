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
import sys
import time
import yaml

from mpi4py import MPI

from ggce.engine.structures import InputParameters
from ggce.engine import system
from ggce.utils.logger import default_logger as _dlog
from ggce.utils import utils


PRINT_EVERY_PERCENT = 20


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


def prep_jobs(k_grid, w_grid, rank, world_size):
    """Prepares the jobs to run by assigning each MPI process a chunk of the
    total job list."""

    jobs = list(product(k_grid, w_grid))
    jobs = [jobs[ii::world_size] for ii in range(world_size)]
    return jobs[rank]


def prime_system(inp):
    t0 = time.time()
    with utils.DisableLogger():
        sy = system.System(inp)
        T = sy.initialize_generalized_equations()
        L = sy.initialize_equations()
        sy.generate_unique_terms()
        sy.prime_solver()
    dt = (time.time() - t0) / 60.0
    return (sy, dt, T, L)


def state_fname(k, w):
    return f"{k:.08f}_{w:.08f}"


def dryrun_random_result():
    """Returns a random value for G, 0.0 for the elapsed time, and 0 for the
    maximum matrix size."""

    G = np.abs(np.random.random()) + np.abs(np.random.random()) * 1j
    return (G, 0.0, 0)


def write_results(
    res_file, k, w, G, elapsed_time, largest_mat_dim, state_fname_path
):
    """Writes the results of a computation to disk."""

    with open(res_file, "a") as f:
        f.write(
            f"{k:.08f}\t{w:.08f}\t{G.real:.08f}\t{G.imag:.08f}"
            f"\t{elapsed_time:.02e}\t{largest_mat_dim}\n"
        )
    with open(state_fname_path, 'w') as f:
        f.write("DONE\n")


def check_state(state_dir, k_u_pi, w):
    """Always returns the state file path, but also checks if the file already
    exists. Returns True if the file does exist, else False."""

    state_fname_path = os.path.join(
        state_dir, state_fname(k_u_pi, w)
    )

    # Never re-run a result if it exists already
    exists = False
    if os.path.isfile(state_fname_path):
        exists = True

    return exists, state_fname_path


def calculate(mpi_info, package_path, config_path, dry_run=False):
    """Runs the calculations.

    Parameters
    ----------
    mpi_info : RankTools
        Helper class containing information about the communicator and contains
        the logger on this rank.
    package_path : str
    config_path : str
        Location of the particular config file to load.
    """

    logger = mpi_info.logger
    rank = mpi_info.RANK
    world_size = mpi_info.SIZE

    # Target directory, will contain res.txt (the results) and STATE, which
    # tracks which calculations were completed.
    base = os.path.splitext(os.path.basename(config_path))[0]
    target_dir = os.path.join(package_path, "results", base)
    res_file = os.path.join(target_dir, "res.txt")
    state_dir = os.path.join(target_dir, "STATE")

    # Check early exit criterion
    donefile = os.path.join(state_dir, "DONE")
    if os.path.isfile(donefile):
        dlog.warning(f"Target {target_dir} is complete")
        return

    # If there are calculations to run, prepare the system
    inp = InputParameters(yaml.safe_load(open(config_path)))
    inp.prime()
    w_grid = inp.get_w_grid()
    k_grid = inp.get_k_grid()  # In units of pi!
    sy = None
    if not dry_run:
        (sy, dt, T, L) = prime_system(inp)
        if rank == 0:
            dlog.info(f"Solver primed with {T}/{L} terms in {dt:.01f}m")
    if dry_run and rank == 0:
        logger.warning("Running in dry run mode: G is randomly generated")

    jobs = prep_jobs(k_grid, w_grid, rank, world_size)

    L = len(jobs)
    print_every = max(L // PRINT_EVERY_PERCENT, 1)

    overall_config_time = time.time()
    for cc, (k_u_pi, frequency_gridpoint) in enumerate(jobs):

        exists, state_fname_path = \
            check_state(state_dir, k_u_pi, frequency_gridpoint)
        if exists:
            logger.debug(f"Target {state_fname_path} exists, continuing")
            continue

        # Solve the system
        if not dry_run:
            with utils.DisableLogger():
                G, meta = sy.solve(k_u_pi * np.pi, frequency_gridpoint)
            A = -G.imag / np.pi
            computation_time = sum(meta['time']) / 60.0
            largest_mat_dim = meta['inv'][0]
            logger.debug(
                f"Solved A({k_u_pi:.02f}pi, {frequency_gridpoint:.02f}) "
                f"= {A:.02f} in {computation_time:.02f}m"
            )
            if A < 0.0:
                logger.error(f"Negative spectral weight: {A:.02e}")
            if (cc % print_every == 0 or cc == 0) and rank == 0:
                logger.info(f"{cc:05}/{L:05} done in {computation_time:.02f}m")
            sys.stdout.flush()

        else:
            G, computation_time, largest_mat_dim = dryrun_random_result()

        # Write results to disk
        write_results(
            res_file, k_u_pi, frequency_gridpoint, G, computation_time,
            largest_mat_dim, state_fname_path
        )

    return jobs, time.time() - overall_config_time


def cleanup(jobs, mpi_info, package_path, config_path):
    """Cleans up the directories by removing specific files that indicate which
    jobs are complete. Mainly, this is done to allow for checkpointing and to
    reduce the number of files when creating a compressed file of the
    results."""

    logger = mpi_info.logger

    # Target directory, will contain res.txt (the results) and STATE, which
    # tracks which calculations were completed.
    base = os.path.splitext(os.path.basename(config_path))[0]
    target_dir = os.path.join(package_path, "results", base)
    state_dir = os.path.join(target_dir, "STATE")

    # Check to make sure there are still w-points to delete in the
    # state directory, this avoids possible race conditions.
    donefile = os.path.join(state_dir, "DONE")
    N_in_state = os.listdir(state_dir)
    if os.path.isfile(donefile) and len(N_in_state) == 1:
        logger.debug(f"Target {target_dir} is done")
        return

    for (k_u_pi, frequency_gridpoint) in jobs:

        state_fname_path = os.path.join(
            state_dir, state_fname(k_u_pi, frequency_gridpoint)
        )
        os.remove(state_fname_path)

    # Add a new file
    with open(donefile, 'a') as f:
        f.write(f"RANK {mpi_info.RANK:05} TAGGED\n")

    logger.debug(f"Confirming target {target_dir} is DONE")


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD  # Default MPI communicator

    # The first argument passed is the base path for the calculation.
    package_path = str(sys.argv[1])

    # The second argument is if to run in debug mode or not
    debug = int(sys.argv[2])

    # The third argument is whether to run in dry run mode or now
    dry_run = int(sys.argv[3])

    # MPI info includes the logger on that rank
    mpi_info = RankTools(COMM, _dlog, debug)
    dlog = mpi_info.logger

    if mpi_info.RANK == 0:
        # Load in the data, which will be used to process into the jobs. The
        # master list contains the list of all omega -> config key pairs (which
        # correspond to the config mapping). The config_mapping maps a number
        # to an actual python dictionary containing the configuration for that
        # trial. The N_M_eta_permutations is a list of the (M, N, eta) to run,
        # and the package_cache_path is the base cache path.
        COMM_timer = time.time()
        all_configs_paths = utils.listdir_fullpath(
            os.path.join(package_path, "configs")
        )
        all_configs_paths.sort()
        dlog.info(f"Confirming COMM world size: {mpi_info.SIZE}")
        dlog.info(f"Running {len(all_configs_paths)} config files")
    else:
        COMM_timer = None
        all_configs_paths = None

    rank_timer = time.time()
    all_configs_paths = COMM.bcast(all_configs_paths, root=0)

    dlog.debug("<- RANK starting up")

    # Iterate over the config files
    jobs_on_config = []
    for ii, config_path in enumerate(all_configs_paths):
        if mpi_info.RANK == 0:
            dlog.info(f"Starting config {ii:03}")
        c_jobs, elapsed = \
            calculate(mpi_info, package_path, config_path, dry_run=dry_run)
        jobs_on_config.append(c_jobs)

        elapsed = COMM.gather(elapsed, root=0)

        if mpi_info.RANK == 0:
            avg = np.mean(elapsed) / 60.0
            sd = np.std(elapsed) / 60.0
            dlog.info(f"Done in {avg:.02f} +/- {sd:.02f} m")

        COMM.Barrier()

    rank_timer_dt = (time.time() - rank_timer) / 3600.0
    dlog.info(f"Done in {rank_timer_dt:.02f}h and waiting for other ranks")

    # Stop all processes here until all complete
    COMM.Barrier()

    for jobs, config_path in zip(jobs_on_config, all_configs_paths):
        cleanup(jobs, mpi_info, package_path, config_path)
    COMM.Barrier()

    if mpi_info.RANK == 0:
        time.sleep(1)
        dt = (time.time() - COMM_timer) / 3600.0
        dlog.info(f"All ranks done in {dt:.02f}h")
