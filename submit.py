#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import copy
import numpy as np
from pathlib import Path
import sys
import time

from mpi4py import MPI

from ggce.engine.structures import GridParams, protocol_mapping
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

    if mpi_info.rank == 0:
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

        mpi_info.logger.info(f"Confirming COMM world size: {mpi_info.size}")
        mpi_info.logger.info(f"Running {len(all_configs_paths)} config files")
        mpi_info.logger.info(f"Will use solver type {solver}")
        mpi_info.logger.info(
            f"Dryrun is {dry_run}; debug is {debug}; buffer is {nbuff}"
        )

        grid_path = package_path / Path("grids.yaml")
        gp = GridParams(grid_path)
        w_grid = gp.get_grid('w')
        k_grid = gp.get_grid('k')  # In units of pi!
        method = gp.grid_info["protocol"]

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
        if mpi_info.rank == 0:
            mpi_info.logger.warning("No configs to run: exiting")
        COMM.Abort()

    if method not in list(protocol_mapping.keys()):
        COMM.Abort()

    # Print some information about the method being run
    if mpi_info.rank == 0:
        if method == 0:
            mpi_info.logger.info("*** Standard calculation")
        elif method == 1:
            mpi_info.logger.info("*** Lowest energy band calculation")

    original_k_grid = k_grid
    original_w_grid = w_grid

    for config_index, config_path in enumerate(all_configs_paths):

        k_grid = copy.deepcopy(original_k_grid)
        w_grid = copy.deepcopy(original_w_grid)

        if mpi_info.rank == 0:
            L = len(all_configs_paths)
            cidx = f"{(config_index + 1):08}"
            mpi_info.logger.info(f"CONFIG: {cidx} / {L:08}")

        # ---------------------------------------------------------------------
        # CALCULATE -----------------------------------------------------------

        # Standard calculation, runs all k points and w points on a grid
        # Is Theta(Nk * Nw) in computational complexity
        if method == "zero temperature":

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
            if mpi_info.rank == 0:
                avg = np.mean(elapsed) / 60.0
                sd = np.std(elapsed) / 60.0
                mpi_info.logger.info(
                    f"CALCULATE done in {avg:.02f} +/- {sd:.02f} m"
                )
            COMM.Barrier()

        # Serial ground state band algorithm, runs all k points. For the first
        # k point, starts with w0. Incremements in steps of eta/10 until
        # the initial peak is found. Once the first peak is found, the next
        # k point starts at eta * 1.5 before the peak, and the process
        # continues.
        elif method == "zero temperature ground state":

            # The w-grid contains special information in the ground state
            # calculation case.
            (w0, w_N_max, eta_div, eta_step_div, next_k_offset_factor) = w_grid

            # We do this method only in serial and for now only on one rank.
            if mpi_info.rank != 0:
                mpi_info.logger.critical(
                    "GS calculation can only be run in serial"
                )
                COMM.Abort()

            # Startup the Executor, which is a helper class for running the
            # calculation using an MPI implementation.
            executor = LowestBandExecutor(
                mpi_info, package_path, config_path, solver, dry_run,
                eta_div, eta_step_div, next_k_offset_factor
            )

            # We prime only once
            executor.prime_calculate()

            # Get the revised grids based on the calculations that have been
            # previously run. There are a few possible options here. Note that
            # there will always be a k-index to resume at, even if it's zero.
            # The k-grid should then iterate from the returned value to the
            # end of the k-grid. The w value is the float to resume at.
            # The results object returned is a concatenated array of all the
            # current results at this k-point (in order of w).
            resume_k_at_idx, w_val, results = executor.recalculate_grids(w0)

            mpi_info.logger.info(f"TODO k-grid of len {len(k_grid)}")

            k_grid = k_grid[resume_k_at_idx:]
            Nk = len(k_grid)

            mpi_info.logger.info(
                f"Resuming at k,w={resume_k_at_idx},{w_val:.02f}"
            )

            k_ii = resume_k_at_idx
            for _, k_val in enumerate(k_grid):

                mpi_info.logger.info(f"k-point: ({k_ii}/{Nk}) k={k_val:.02f}")

                current_n_w = 0
                reference = 0.0
                while True:

                    if current_n_w > w_N_max:
                        executor.finalize()
                        mpi_info.logger.critical(
                            "Exceeded maximum omega points. Aborting."
                        )
                        COMM.Abort()

                    # Run the calculation (note k is in units of pi and is
                    # converted in executor.calculate).
                    val, elapsed = executor.calculate(k_val, w_val)
                    spectrum = -val[3] / np.pi
                    mpi_info.logger.info(
                        f"Done ({current_n_w}) w={w_val:.05f}, "
                        f"A={spectrum:.05f} in "
                        f"{(elapsed/60.0):.02f} m"
                    )
                    results.append(val)

                    # Check and see whether or not we've found a local maxima
                    if reference < spectrum:

                        # This is not a maximum
                        reference = spectrum

                        # Save the result to disk
                        executor.save(k_ii, results)

                        current_n_w += 1
                        w_val += executor.eta_step
                        continue

                    # This is a maximum. Run the calculation using
                    # eta prime
                    val2, elapsed = executor.calculate(
                        k_val, w_val, use_eta_prime=True
                    )
                    spectrum_prime = -val2[3] / np.pi
                    mpi_info.logger.info(
                        f"Done ({current_n_w}) w={w_val:.05f}, "
                        f"A={spectrum_prime:.05f} in "
                        f"{(elapsed/60.0):.02f} m"
                    )

                    # Get the peak location and QP weight exactly
                    loc, weight = executor.peak_location_and_weight(
                        w_val, spectrum, spectrum_prime
                    )

                    mpi_info.logger.info(
                        f"Found peak (loc, weight)=({loc:.05e}, {weight:.02e})"
                    )

                    # Save the result to disk
                    executor.save(k_ii, results, final=[val2, loc, weight])

                    # Estimate the new w-point to calculate for the next
                    # k value
                    w_val = loc - executor.inp.eta * next_k_offset_factor

                    # Continue with the next k-point
                    results = []
                    break

                k_ii += 1

        if method == "zero temperature ground state":
            executor.finalize()

        elif method == "zero temperature":

            # -----------------------------------------------------------------
            # FINALIZE --------------------------------------------------------

            # Let the 0th rank list all files in the current config directory
            # and scatter them to the respective ranks.
            if mpi_info.rank == 0:
                tmp_t = time.time()
                state_files = utils.listdir_fullpath(executor.state_dir)
                state_files = mpi_info.chunk_jobs(state_files)
            else:
                state_files = None
            state_files = COMM.scatter(state_files, root=0)

            # Begin the concatenation process of collecting all of the STATE
            # files
            res = executor.finalize(state_files)
            res = COMM.gather(res, root=0)

            if mpi_info.rank == 0:
                tmp_t = (time.time() - tmp_t) / 60.0
                mpi_info.logger.info(f"FINALIZE done in {tmp_t:.02f} m")

                # Concatenate the results on rank 0 and save to disk
                res = [r for r in res if r is not None]
                res = np.concatenate(res, axis=0)
                executor.save_final(res)
            COMM.Barrier()

            # -----------------------------------------------------------------
            # CLEANUP ---------------------------------------------------------

            if mpi_info.rank == 0:
                tmp_t = time.time()

            executor.cleanup(state_files)
            COMM.Barrier()

            # Last step is to delete STATE
            if mpi_info.rank == 0:
                Path(executor.state_dir).rmdir()

            if mpi_info.rank == 0:
                tmp_t = (time.time() - tmp_t) / 60.0
                mpi_info.logger.info(f"CLEANUP done in {tmp_t:.02f} m")
            COMM.Barrier()

        COMM.Barrier()
        if mpi_info.rank == 0:
            time.sleep(1)
            dt = (time.time() - COMM_timer) / 3600.0
            mpi_info.logger.info(f"ALL done in {dt:.02f} h")
