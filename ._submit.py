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

    job_lens = [len(j) for job_part in jobs for j in list(job_part.values())]

    # for ii in range(len(jobs)):
    #     for cc in list(jobs[ii].keys()):
    #         job_lens.append(jobs[ii][cc])
    # dlog.warning(
    #     f"Possible job imbalances at e.g. processes {ii}/{ii + 1} "
    #     f"with lens {len(jobs[ii][cc])} and "
    #     f"{len(jobs[ii + 1][cc])}"
    # )

    job_lens = np.array(job_lens)
    mean_job_number = np.mean(job_lens)
    std_job_number = np.std(job_lens)
    max_job = np.max(job_lens)
    min_job = np.min(job_lens)

    if std_job_number > 0.0:
        dlog.warning(
            f"Job imbalance: {mean_job_number:.02f} +/- "
            f"{std_job_number:.02f} jobs/rank (min/max {min_job}/{max_job})"
        )

    return jobs


def prime_system(M, N, eta, config, logger):
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
    logger.info(f"{M},{N},{eta:.02e} terms {T} & {L} primed in {dt:.02f}m")
    return sy


def log_status(logger, M, N, eta, k, w, dt_wgrid_final, cc, total_points):
    logger.info(
        f"Combination {M},{N},{eta:.02e} "
        f"done with {w:.02f}/{k:.02f} in {dt_wgrid_final:.02f}h "
        f"({cc:05}/{total_points:05})"
    )


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

    if dry_run and logger.rank == 0:
        logger.warning("Running in dry run mode: G is randomly generated")

    perms = list(product(
        M_N_eta_k_mapping['M'], M_N_eta_k_mapping['N'],
        M_N_eta_k_mapping['eta']
    ))

    k_to_calculate = M_N_eta_k_mapping['k_units_pi']

    for c_idx, wgrid in jobs.items():
        config = config_mapping[c_idx]

        for perm in perms:
            (M, N, eta) = perm

            # For a given permutation of M, N and eta, we check to see if
            # for every k-point and every w-point, the calculation has been
            # completed. This also serves the purpose of possibly making
            # necessary sub-directories.
            all_done = True
            for k_u_pi in k_to_calculate:
                target = utils.N_M_eta_k_subdir(
                    *perm, k_u_pi, M_N_eta_k_mapping, c_idx
                )
                target = os.path.join(package_cache_path, target)
                state_dir = os.path.join(target, 'state')
                donefile = os.path.join(state_dir, "DONE.txt")
                if not os.path.isfile(donefile):
                    dlog.debug(f"Target {target} is not complete")
                    all_done = False

            if all_done:
                continue

            # It not all done, prime the system. We only do this once per
            # (M, N, eta) so as to save time.
            if not dry_run:
                sy = prime_system(M, N, eta, config, logger)

            total_points = len(k_to_calculate) * len(wgrid)
            print_every = total_points // 10
            cc = 1
            for k_u_pi in k_to_calculate:
                for w in wgrid:
                    wgrid_t0 = time.time()
                    target = utils.N_M_eta_k_subdir(
                        *perm, k_u_pi, M_N_eta_k_mapping, c_idx
                    )
                    target = os.path.join(package_cache_path, target)
                    os.makedirs(target, exist_ok=True)
                    state_dir = os.path.join(target, 'state')
                    os.makedirs(state_dir, exist_ok=True)
                    state_path = os.path.join(state_dir, f"w_{w:.12f}.txt")

                    # Never re-run a result if it exists already
                    if os.path.isfile(state_path):
                        logger.warning(
                            f"Target {state_path} exists, continuing"
                        )
                        continue

                    logger.debug(
                        f"Running {M},{N},{eta:.02e},{k_u_pi:.02f}, w={w:.12f}"
                    )

                    # Solve the system
                    if not dry_run:
                        t0 = time.time()
                        with utils.DisableLogger():
                            G, meta = sy.solve(k_u_pi * np.pi, w)
                        dt = (time.time() - t0) / 60.0
                        A = -G.imag / np.pi
                        logger.debug(
                            f"Solved A({k_u_pi:.02f}pi, {w:.02f}) "
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
                            f"{w:.08f}\t{G.real:.08f}\t{G.imag:.08f}\t{t:.02e}"
                            f"\t{largest_mat_dim}\n"
                        )
                    with open(state_path, 'w') as f:
                        f.write("DONE\n")

                    dt_wgrid_final = (time.time() - wgrid_t0) / 3600.0
                    if cc % print_every == 0 or cc == 1:
                        log_status(
                            logger, M, N, eta, k_u_pi, w,
                            dt_wgrid_final, cc, total_points
                        )
                    cc += 1


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

            # Check to make sure there are still w-points to delete in the
            # state directory, this avoids possible race conditions.
            donefile = os.path.join(state_dir, "DONE.txt")
            N_in_state = os.listdir(state_dir)
            if os.path.isfile(donefile) and len(N_in_state) == 1:
                dlog.debug(f"Target {target} is done")
                continue

            for w in wgrid:
                state_path = os.path.join(state_dir, f"w_{w:.12f}.txt")
                os.remove(state_path)

            # Add a new file
            with open(donefile, 'a') as f:
                f.write(f"RANK {logger.rank:05} TAGGED\n")

            logger.debug(f"Confirming target {target} is DONE")


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD  # Default MPI communicator
    RANK = COMM.rank

    # The first argument passed is the base path for the calculation.
    base_path = str(sys.argv[1])

    # The second argument is if to run in debug mode or not
    debug = int(sys.argv[2])

    # The third argument is whether to run in dry run mode or now
    dry_run = int(sys.argv[3])

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
        dlog.info(f"Confirming COMM world size: {COMM.size}")
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

    dlog.debug("<- RANK starting up")
    dlog.debug(f"Received vars incl. {package_cache_path}")

    calculate(
        master_dict, config_mapping, M_N_eta_k_mapping, package_cache_path,
        dlog, dry_run=dry_run
    )
    rank_timer_dt = (time.time() - rank_timer) / 3600.0
    dlog.info(f"Done in {rank_timer_dt:.02f}h and waiting for other ranks")

    # Stop all processes here until all complete
    COMM.Barrier()

    # Erase the state directory files, and write DONE to the correct
    # locations
    cleanup(
        master_dict, M_N_eta_k_mapping, package_cache_path, dlog,
        dry_run=dry_run
    )

    COMM.Barrier()

    if RANK == 0:
        time.sleep(1)
        dt = (time.time() - COMM_timer) / 3600.0
        dlog.info(f"All ranks done in {dt:.02f}h")
