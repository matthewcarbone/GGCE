#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import numpy as np
import multiprocessing as mp
import os
import time

from ggce import system
from ggce.utils.logger import default_logger as dlog


def execute(k, w_arr, sy, log_every):
    """Function for running on a single process."""

    results = []
    L = len(w_arr)
    pid = os.getpid()
    for ii, w in enumerate(w_arr):
        t0 = time.time()
        G, meta = sy.solve(k, w)
        results.append([w, G, meta])
        if (ii + 1) % log_every == 0:
            pc = (ii + 1) / L * 100.0
            dt = time.time() - t0
            dlog.info(
                f"({pid}, {pc:.01f}%, {dt:.01f}s) done A({k:.02f}, {w:.02f})"
            )

    return results


def parallel(
    k, w_arr, w_bins, config, nprocs=mp.cpu_count() - 1,
    log_every=50
):
    """Runs in parallel.

    Parameters
    ----------
    k : float
        The value for the momentum.
    w_arr : array_like
        Array containing the desired w points.
    w_bins : int
        The number of bins for separating the w-points in the multiprocessing
        framework.
    config : ggce.structures.InputParameters
        The configuration for the trial.
    nprocs : int
        The number of processes in the multiprocessing pool. Each proc in the
        pool will end up being assigned OMP_NUM_THREADs threads. So ultimately,
        the number of processes * the number of threads should approximately
        equal the total number of available CPU's.
    """

    t0 = time.time()

    sy = system.System(config)
    sy.initialize_generalized_equations()
    sy.initialize_equations()
    sy.generate_unique_terms()
    sy.prime_solver()

    threads = os.environ.get("OMP_NUM_THREADS")
    if threads != "1":
        dlog.warning(f"OMP_NUM_THREADS ({threads}) != 1")

    w_arrays = np.array_split(w_arr, w_bins)

    pool = mp.Pool(nprocs)

    processes = []
    for w in w_arrays:
        processes.append(pool.apply_async(execute, args=(k, w, sy, log_every)))
    pool.close()
    pool.join()

    results = [l for p in processes for l in p.get()]
    results.sort(key=lambda x: x[0])  # Sort by w
    G = np.array([x[1] for x in results])
    meta = [x[2] for x in results]

    dt = time.time() - t0
    dlog.info(f"({dt:.02f}s) Parallel execution complete")
    return G, meta
