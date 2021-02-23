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

import copy
import numpy as np
from pathlib import Path
import sys
import time
import yaml

from mpi4py import MPI

from ggce.engine.structures import GridParams
from ggce.executors import Executor, LowestBandExecutor
from ggce.utils.logger import default_logger as _dlog
from ggce.utils import utils


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD  # Default MPI communicator

    # The first argument passed is the base path for the calculation.
    package_path = Path(str(sys.argv[1]))

    # The second argument is if to run in debug mode or not
    debug = int(sys.argv[2])

    # The third argument is whether to run in dry run mode or now
    dry_run = int(sys.argv[3])

    # Type of solver
    solver = int(sys.argv[4])

    # Number of calculation steps before flushing the buffer. Default CL arg
    # is -1 corresponding to int(max(calculations // 100, 1)).
    nbuff = int(sys.argv[5])

    # MPI info includes the logger on that rank
    mpi_info = utils.RankTools(COMM, _dlog, debug)

    if mpi_info.RANK == 0:
        COMM_timer = time.time()
        configs_path = package_path / Path("configs")
        results_path = package_path / Path("results")
        _all_configs_paths = utils.listdir_fullpath(configs_path)

        # Remove any config with a donefile
        all_configs_paths = []
        for config in _all_configs_paths:
            done_file = results_path / Path(config.stem) / Path("DONE")
            if not done_file.exists():
                all_configs_paths.append(config)

        all_configs_paths.sort()

        mpi_info.logger.info(f"Confirming COMM world size: {mpi_info.SIZE}")
        mpi_info.logger.info(f"Running {len(all_configs_paths)} config files")
        mpi_info.logger.info(f"Will use solver type {solver}")
        mpi_info.logger.info(
            f"Dryrun is {dry_run}; debug is {debug}; buffer is {nbuff}"
        )

        grid_path = package_path / Path("grids.yaml")
        gp = GridParams(yaml.safe_load(open(grid_path)))
        w_grid = gp.get_grid('w')
        k_grid = gp.get_grid('k')  # In units of pi!
        method = gp.method

    else:
        COMM_timer = None
        all_configs_paths = None
        w_grid = None
        k_grid = None
        method = None

    rank_timer = time.time()

    # Iterate over the config files
    all_configs_paths = COMM.bcast(all_configs_paths, root=0)
    w_grid = COMM.bcast(w_grid, root=0)
    k_grid = COMM.bcast(k_grid, root=0)
    method = COMM.bcast(method, root=0)

    if all_configs_paths == []:
        if mpi_info.RANK == 0:
            mpi_info.logger.warning("No configs to run: exiting")
        COMM.Abort()

    if method not in ['standard', 'gs']:
        COMM.Abort()

    # Print some information about the method being run
    if mpi_info.RANK == 0:
        if method == 0:
            mpi_info.logger.info("*** Standard calculation")
        elif method == 1:
            mpi_info.logger.info("*** Lowest energy band calculation")

    original_k_grid = k_grid
    original_w_grid = w_grid

    for config_index, config_path in enumerate(all_configs_paths):

        k_grid = copy.deepcopy(original_k_grid)
        w_grid = copy.deepcopy(original_w_grid)

        if mpi_info.RANK == 0:
            L = len(all_configs_paths)
            cidx = f"{(config_index + 1):08}"
            mpi_info.logger.info(f"CONFIG: {cidx} / {L:08}")

        # ---------------------------------------------------------------------
        # CALCULATE -----------------------------------------------------------

        # Standard calculation, runs all k points and w points on a grid
        # Is Theta(Nk * Nw) in computational complexity
        if method == 'standard':

            # Startup the Executor, which is a helper class for running the
            # calculation using an MPI implementation
            executor = Executor(
                mpi_info, package_path, config_path, solver, dry_run, nbuff
            )

            # Run the calculation on this rank
            elapsed = executor.calculate(k_grid, w_grid)

            # Collect the runtimes for each of these processes. This also
            # serves # as a barrier.
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

        # Ground state band algorithm, runs all k points but uses the w grid
        # as an initial guess for the location of the ground state band, then
        # finds that peak and traces it through k-space. Will re-approximate
        # the w grid at ever iteration.
        elif method == 'gs':

            # Startup the Executor, which is a helper class for running the
            # calculation using an MPI implementation
            executor = LowestBandExecutor(
                mpi_info, package_path, config_path, solver, dry_run, nbuff
            )

            # We prime only once
            executor.prime_calculate()

            # Re-mask the k-grid and re-calculate the w_grid if necessary
            if mpi_info.RANK == 0:
                mpi_info.logger.info(f"Original k-grid of len {len(k_grid)}")
                k_grid, w_grid = executor.recalculate_k_grid(k_grid, w_grid)
                mpi_info.logger.info(f"TODO k-grid of len {len(k_grid)}")

            else:
                k_grid = None
                w_grid = None

            k_grid = COMM.bcast(k_grid, root=0)
            w_grid = COMM.bcast(w_grid, root=0)

            Nk = len(k_grid)

            for k_ii, k_val in enumerate(k_grid):

                if mpi_info.RANK == 0:
                    mpi_info.logger.info(
                        f"k ({k_ii}/{Nk}): {k_val:.02f} w: ({w_grid[0]:.02f} "
                        f"-> {w_grid[-1]:.02f})"
                    )

                # Run the calculation on this rank, for only this one k point
                result, elapsed = executor.calculate(k_val, w_grid)
                COMM.Barrier()

                # Collect the runtimes for each of these processes. This also
                # serves # as a barrier.
                elapsed = COMM.gather(elapsed, root=0)

                # Print some useful information about how fast the overall
                # process was and how imbalanced the loads were
                if k_ii == 0:
                    if mpi_info.RANK == 0:
                        avg = np.mean(elapsed) / 60.0
                        sd = np.std(elapsed) / 60.0
                        mpi_info.logger.info(
                            f"CALCULATE done in {avg:.02f} +/- {sd:.02f} m"
                        )
                COMM.Barrier()

                # Gather all of the results
                result = COMM.gather(result, root=0)

                # Concatenate and save this result
                if mpi_info.RANK == 0:
                    result = [r for r in result if r is not None]
                    result = np.concatenate(result, axis=0)

                    # Save the result to disk
                    executor.save(result)

                    # Get the new w_grid estimate
                    w_grid = executor.gs_peak(result)

                else:
                    w_grid = None

                # Broadcast the new w_grid to all ranks
                w_grid = COMM.bcast(w_grid, root=0)

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
            res = [r for r in res if r is not None]
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
            Path(executor.state_dir).rmdir()

        if mpi_info.RANK == 0:
            tmp_t = (time.time() - tmp_t) / 60.0
            mpi_info.logger.info(f"CLEANUP done in {tmp_t:.02f} m")
        COMM.Barrier()

    COMM.Barrier()
    if mpi_info.RANK == 0:
        time.sleep(1)
        dt = (time.time() - COMM_timer) / 3600.0
        mpi_info.logger.info(f"ALL done in {dt:.02f} h")
