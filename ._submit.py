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

from mpi4py import MPI

from ggce.engine.structures import InputParameters
from ggce.engine import system
from ggce.utils.logger import default_logger as _dlog
from ggce.utils import utils

DRY_RUN = False


class LoggerOnRank:

    def __init__(self, rank, logger, debug_flag=False):
        self.rank = rank
        self.logger = logger
        self.debug_flag = debug_flag

    def debug(self, msg):
        if self.debug_flag:
            self.logger.debug(f"({self.rank}) {msg}")

    def info(self, msg):
        self.logger.info(f"({self.rank}) {msg}")

    def warning(self, msg):
        self.logger.warning(f"({self.rank}) {msg}")

    def error(self, msg):
        self.logger.error(f"({self.rank}) {msg}")

    def critical(self, msg):
        self.logger.critical(f"({self.rank}) {msg}")


def prep_jobs(master_dict, logger, comm):
    """Prepares the jobs to run by assigning each MPI process a chunk of the
    total job list.
        The jobs object is a dictionary with keys indexing the config and
    values the frequency points. Each of these should be split as equally as
    possible such that the jobs are balanced in w.
    """

    jobs = [
        {
            cc: wpts[ii::comm.size]
            for cc, wpts in master_dict.items()
        } for ii in range(comm.size)
    ]

    for ii in range(len(jobs) - 1):
        for cc in list(jobs[ii].keys()):
            if len(jobs[ii][cc]) != len(jobs[ii + 1][cc]):
                dlog.warning(
                    f"Possible job imbalances at e.g. processes {ii}/{ii + 1} "
                    f"with lens {len(jobs[ii][cc])} and "
                    f"{len(jobs[ii + 1][cc])}"
                )
                return jobs

    return jobs


def calculate(
    jobs, config_mapping, M_N_eta_k_mapping, package_cache_path, logger,
    dry_run=False
):
    """Runs the calculations.

    Parameters
    ----------
    jobs : dict
        The jobs to run on this rank.
    config_mapping : dict
        A map between the config index and the actual config. Note that the
        k-values here are still in units of pi.
    M_N_eta_k_mapping : dict
        A master list of the keys M, N, eta and k, and the values mapped
        to the indexes we will use to construct the directory paths.
    package_cache_pat : str
        The location of the cache for this package.
    """

    if dry_run:
        logger.warning("Running in dry run mode: G is randomly generated")

    perms = list(product(
        M_N_eta_k_mapping['M'], M_N_eta_k_mapping['N'],
        M_N_eta_k_mapping['eta'], M_N_eta_k_mapping['k_units_pi']
    ))

    for c_idx, wgrid in jobs.items():
        config = config_mapping[c_idx]

        for perm in perms:
            (M, N, eta, k_units_pi) = perm
            target = utils.N_M_eta_k_subdir(*perm, M_N_eta_k_mapping, c_idx)
            target = os.path.join(package_cache_path, target)
            state_dir = os.path.join(target, 'state')
            os.makedirs(state_dir, exist_ok=True)
            os.makedirs(target, exist_ok=True)

            donefile = os.path.join(state_dir, "DONE.txt")
            if os.path.isfile(donefile):
                dlog.debug(f"Target {target} is done, continuing")
                continue

            if not dry_run:
                input_params = InputParameters(
                    M=M, N=N, eta=eta, t=config['t'], Omega=config['Omega'],
                    lambd=config['lambda'], model=config['model'],
                    config_filter=config['config_filter']
                )
                logger.debug(input_params.get_params())
                input_params.init_terms()

                # Prime the system
                t0 = time.time()
                with utils.DisableLogger():
                    sy = system.System(input_params)
                    T = sy.initialize_generalized_equations()
                    L = sy.initialize_equations()
                    sy.generate_unique_terms()
                    sy.prime_solver()
                dt = (time.time() - t0) / 60.0
                logger.info(
                    f"{M},{N},{eta:.02e},{k_units_pi:.02f} terms {T} & {L} "
                    f"primed in {dt:.02f}m"
                )

            for w in wgrid:
                state_path = os.path.join(state_dir, f"w_{w:.12f}.txt")

                # Never re-run a result if it exists already
                if os.path.isfile(state_path):
                    logger.warning(f"Target {state_path} exists, continuing")
                    continue

                logger.debug(
                    f"Running {M},{N},{eta:.02e},{k_units_pi:.02f}, w={w:.12f}"
                )

                # Solve the system
                if not dry_run:
                    t0 = time.time()
                    with utils.DisableLogger():
                        G, meta = sy.solve(k_units_pi * np.pi, w)
                    dt = (time.time() - t0) / 60.0
                    A = -G.imag / np.pi
                    logger.info(
                        f"Solved A({k_units_pi:.02f}pi, {w:.02f}) "
                        f"= {A:.02f} in {dt:.02f}m"
                    )
                    if A < 0.0:
                        logger.error(f"Negative spectral weight: {A:.02e}")
                    t = sum(meta['time'])
                    largest_mat_dim = meta['inv'][0]
                    sys.stdout.flush()

                else:
                    G = np.abs(np.random.random()) + \
                        np.abs(np.random.random()) * 1j
                    t = 0.0
                    largest_mat_dim = 0

                # Write results to disk
                with open(os.path.join(target, 'res.txt'), "a") as f:
                    f.write(
                        f"{w:.12f}\t{G.real:.12f}\t{G.imag:.12f}\t{t:.02e}"
                        f"\t{largest_mat_dim}\n"
                    )
                with open(state_path, 'w') as f:
                    f.write("DONE\n")


def cleanup(
    jobs, M_N_eta_k_mapping, package_cache_path, logger,
    dry_run=False
):
    """Runs the calculations.

    Parameters
    ----------
    jobs : dict
        The jobs to run on this rank.
    M_N_eta_k_mapping : dict
        A master list of the keys M, N, eta and k, and the values mapped
        to the indexes we will use to construct the directory paths.
    package_cache_pat : str
        The location of the cache for this package.
    """

    perms = list(product(
        M_N_eta_k_mapping['M'], M_N_eta_k_mapping['N'],
        M_N_eta_k_mapping['eta'], M_N_eta_k_mapping['k_units_pi']
    ))

    for c_idx, wgrid in jobs.items():
        for perm in perms:
            (M, N, eta, k_units_pi) = perm
            target = utils.N_M_eta_k_subdir(*perm, M_N_eta_k_mapping, c_idx)
            target = os.path.join(package_cache_path, target)
            state_dir = os.path.join(target, 'state')

            donefile = os.path.join(state_dir, "DONE.txt")
            if os.path.isfile(donefile):
                dlog.debug(f"Target {target} is done")
                continue

            for w in wgrid:
                state_path = os.path.join(state_dir, f"w_{w:.12f}.txt")
                os.remove(state_path)

            # Add a new file
            with open(donefile, 'a') as f:
                f.write(f"RANK {logger.rank} TAGGED\n")

            logger.debug(f"Confirming target {target} is DONE")


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD  # Default MPI communicator
    RANK = COMM.rank

    # The first argument passed is the base path for the calculation.
    base_path = str(sys.argv[1])

    # The second argument is if to run in debug mode or not
    debug = int(sys.argv[2])

    dlog = LoggerOnRank(rank=RANK, logger=_dlog, debug_flag=bool(debug))

    if RANK == 0:
        # Load in the data, which will be used to process into the jobs. The
        # master list contains the list of all omega -> config key pairs (which
        # correspond to the config mapping). The config_mapping maps a number
        # to an actual python dictionary containing the configuration for that
        # trial. The N_M_eta_permutations is a list of the (M, N, eta) to run,
        # and the package_cache_path is the base cache path.
        COMM_timer = time.time()
        (
            master_mapping, config_mapping,
            M_N_eta_k_mapping, package_cache_path
        ) = pickle.load(open(os.path.join(base_path, 'protocol.pkl'), 'rb'))
        dlog.debug(f"Loaded jobs from {base_path}")
        master_dict = prep_jobs(master_mapping, dlog, COMM)
        dlog.debug(f"Cache path is {package_cache_path}")
        for key, value in M_N_eta_k_mapping.items():
            dlog.info(f"Running {key}, incl. {list(value.keys())[:4]}")
    else:
        COMM_timer = None
        master_dict = None
        config_mapping = None
        M_N_eta_k_mapping = None
        package_cache_path = None

    rank_timer = time.time()

    # Scatter jobs across cores.
    master_dict = COMM.scatter(master_dict, root=0)

    # Broadcast the other variables
    config_mapping = COMM.bcast(config_mapping, root=0)
    assert config_mapping is not None
    M_N_eta_k_mapping = COMM.bcast(M_N_eta_k_mapping, root=0)
    assert M_N_eta_k_mapping is not None
    package_cache_path = COMM.bcast(package_cache_path, root=0)
    assert package_cache_path is not None

    dlog.info("<- RANK starting up")
    dlog.debug(f"Received vars incl. {package_cache_path}")

    calculate(
        master_dict, config_mapping, M_N_eta_k_mapping, package_cache_path,
        dlog, dry_run=DRY_RUN
    )
    rank_timer_dt = (time.time() - rank_timer) / 3600.0
    dlog.info(f"Done in {rank_timer_dt:.02f}h and waiting for other ranks")

    # Stop all processes here until all complete
    COMM.Barrier()

    # Erase the state directory files, and write DONE to the correct
    # locations
    cleanup(
        master_dict, M_N_eta_k_mapping, package_cache_path, dlog,
        dry_run=DRY_RUN
    )

    COMM.Barrier()

    if RANK == 0:
        dt = (time.time() - COMM_timer) / 3600.0
        dlog.info(f"All ranks done in {dt:.02f}h")
