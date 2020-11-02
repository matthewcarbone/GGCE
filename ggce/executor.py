#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


from itertools import product
import numpy as np
import multiprocessing as mp
import os
import time
import yaml

from ggce.structures import InputParameters
from ggce import system
from ggce.utils.logger import default_logger as dlog
from ggce.utils import utils


def execute(k, w_arr, sy, log_every, target_dir):
    """Function for running on a single process."""

    results = []
    L = len(w_arr)
    pid = os.getpid()
    for ii, w in enumerate(w_arr):

        state_name = f"{target_dir}/state/{w:.06f}.txt"

        if target_dir is not None:
            if os.path.exists(state_name):
                dlog.debug(f"File {state_name} exists, continuing")
                continue

        t0 = time.time()
        G, meta = sy.solve(k, w)

        if target_dir is None:
            results.append([w, G, meta])
        else:
            with open(f"{target_dir}/res.txt", "w") as f:
                A = -G.imag / np.pi
                t = sum(meta['times'])
                f.write(f"{w}\t{A}\t{t}\n")
            with open(state_name, 'w') as f:
                f.write("DONE\n")

        if (ii + 1) % log_every == 0:
            pc = int((ii + 1) / L * 100.0)
            dt = time.time() - t0
            dlog.info(
                f"({pid}, {pc:03}%, {dt:.01f}s) done A({k:.02f}, {w:.02f})"
            )

    if target_dir is None:
        return results


def parallel(
    w_arr, config, w_bins=-1, nprocs=mp.cpu_count() - 1, log_every=50,
    target_dir=None
):
    """Runs in parallel.

    Parameters
    ----------
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

    w_arrays = np.array_split(w_arr, w_bins if w_bins > 0 else len(w_arr))

    pool = mp.Pool(nprocs)

    processes = []
    k = config.k
    for w in w_arrays:
        processes.append(pool.apply_async(
            execute, args=(k, w, sy, log_every, target_dir)
        ))
    pool.close()
    pool.join()

    results = [l for p in processes for l in p.get()]
    dt = time.time() - t0
    dlog.info(f"({dt:.02f}s) Parallel execution complete")

    if target_dir is not None:
        return

    results.sort(key=lambda x: x[0])  # Sort by w
    G = np.array([x[1] for x in results])
    meta = [x[2] for x in results]
    return G, meta


class Base:

    def __init__(self, args):

        # The directories important for everything we do here.
        self.cache_dir = utils.get_cache_dir()
        self.package_dir = utils.get_package_dir()
        self.args = args


class Submitter(Base):
    def __init__(self, args):
        super().__init__(args)
        dlog.info("Running executor")

    def run(self):
        """Submits all primed trials."""

        sub_args = f"{int(self.args.debug)} {self.args.w_bins}"

        all_cache = utils.listdir_fullpath(self.cache_dir)
        all_cache.sort()
        for ii, package in enumerate(all_cache):

            all_package = utils.listdir_fullpath(package)
            for config in all_package:

                all_config = utils.listdir_fullpath(config)
                all_config = [c for c in all_config if os.path.isdir(c)]
                for dist in all_config:

                    all_dist = utils.listdir_fullpath(dist)

                    # Permutations over parameters
                    for perm in all_dist:

                        submit_script = f"{perm}/script.sh"
                        grid = np.loadtxt(f"{perm}/grid.txt")
                        L = len(grid)
                        completed = os.listdir(f"{perm}/state")
                        Lc = len(completed)
                        pc = Lc / L * 100.0

                        if pc < 100.0:
                            tmp = perm.split("/")
                            tmp = "/".join(tmp[-4:])
                            dlog.info(
                                f"{tmp} is {pc:.02f}% complete: submitting"
                            )
                            utils.run_command(f"mv {submit_script} .")
                            # Path is already contained in the written
                            # submit.sh
                            utils.run_command(f"bash script.sh {sub_args}")
                            utils.run_command(f"mv script.sh {perm}")
                        else:
                            dlog.info(
                                f"{perm} is {pc:.02f} complete, continuing"
                            )
                            continue


class Primer(Base):
    """Intelligently detects existing directories in the cache and plans the
    future computations.
        The directory structure for the priming step is designed as follows.
    In the $GGCE_CACHE_DIR, there are directories, each of which represents
    a config file. Packages are simply collections of config files which can
    all be run at once. When plotting results, an entire packages is read in
    at the same time. This package can consist of one, or many config files.
    Each config file has the following properties:
        * Exactly one value for the model type 'H', 'EFB' or 'SSH'.
        * Exactly one value for lambda, one value for Omega and one value for
          t, the fermion-boson coupling, Einstein boson frequency, and hopping
          terms, respectively.
        * User-desired number of terms for k (in units of pi). Or, the user
          can set the linspacek flag to True, and specify the linspace
          parameters for generating the grid over k.
        * User-desired number of lists for the omega grid. Each element
          of the list represents a linspace chunk over that range.
    Note that the values specified in the config are *not* parameters that we
    converge our calculations with respect to. Those, meaning M, N and eta, are
    provided via the command line. Thus in a sense, the config files specifc
    all non-convergence parameters, and the command line specifies all
    convergence parameters. An example of a cache directory structure is as
    follows, and example config setups are provided as a template in the
    packages directory.

    0000_package_1
    0001_package_2
    |--- config 1
    |--- config 2
    |--- ...
    |--- config N
         |--- 000: (k, N, M, eta) permutation 1
         |--- 001: (k, N, M, eta) permutation 2
         |--- ...
         |--- 00M: (k, N, M, eta) permutation M
         |    |--- Distribution index 0  # Calc was dist. accross many nodes
         |    |--- Distribution index 1
         |    |--- ...
         |    |--- Distribution index P
         |         |--- grid.txt
         |         |--- res.txt
         |         |--- state
         |              |--- idx1.txt  # State of w-point 1
         |              |--- idx2.txt  # State of w-point 2
         |              |--- ...
         |--- config.yaml  # A reproduction of the configuration file
         |--- mapping.yaml  # A map between the directory name and permutations
    - ...

    Attributes
    ----------
    all_packages : list
        A list of the package directory full paths in the $GGCE_PACKAGE_DIR
        (or `packages`) location.
    current_index : int
        The index at which to resume calculations.
    existing_names : list
        A list of strings containing the names of the calculations previous
        run. These will be compared to the current packages to be submitted,
        and warnings will be thrown if they already exist, leading to possibly
        a duplicate calculation.
    """

    def __init__(self, args):
        super().__init__(args)

        dlog.info("Initializing primer")

        # List all of the packages that exist in the package directory (with
        # full paths)
        all_packages = utils.listdir_fullpath(self.package_dir)

        # Exclude the template package, of course
        all_packages = [
            p for p in all_packages
            if ("TEMPLATE" not in p and ".gitignore" not in p)
        ]

        dlog.info(
            f"Found {len(all_packages)} packages including e.g. "
            f"{all_packages[0]}"
        )
        self.all_packages = all_packages

        # List all of the directories already existing in the cache (with full
        # paths)
        existing = utils.listdir_fullpath(self.cache_dir)
        existing.sort()

        # Get the basename of the last file and add one to the index. This
        # marks the index to resume creating directories at.
        if len(existing) > 0:
            dlog.warning(f"Found {len(existing)} dirs already in the cache")
            dlog.info(f"The last of these is {existing[-1]}")
            basename = os.path.basename(existing[-1])
            self.current_index = int(basename.split("_")[0]) + 1
            dlog.info(f"Ready to resume at index {self.current_index}")
        else:
            self.current_index = 0
            dlog.info(f"No existing cache directories, starting at 000")

        if self.current_index > 999:
            dlog.warning("Current index > 999, expect unexpected behavior")

        # Get all the names of the files to compare with those ready to
        # submit later on
        self.existing_names = [
            os.path.basename(dd).split("_")[1]
            for dd in existing
        ]

    def _compute_overall_grid(self, disk_config):
        """Stitches the linspace params together to produce the actual grid.
        Also determines the number of grids to partition the full w-grid into
        depending on the user-supplied distribute command line argument.

        Parameters
        ----------
        disk_config : dict
            Yaml file loaded from disk outlining the parameters of the
            trial(s).

        Returns
        -------
        grids : list
            A list of numpy arrays each of which is one grid, each for one
            compute node.
        """

        d = self.args.distribute
        lp = disk_config['linspace_params']
        assert d > 0
        assert isinstance(d, int)
        grid = np.sort(np.concatenate([
            np.linspace(*c, endpoint=False) for c in lp
        ]))
        grids = [grid]
        if d > 1:
            grids = [
                np.array([
                    grid[ii + xx] for ii in range(0, len(grid), d)
                    if ii + xx < len(grid)
                ]) for xx in range(d)
            ]
        assert sum([len(xx) for xx in grids])
        assert(
            len(np.unique([item for sublist in grids for item in sublist]))
            == len(grid)
        )
        return grids

    def _ready_config(self, base, config, slurm_path):
        """Prepares a single config for running.

        Parameters
        ----------
        base : str
            The path to the directory corresponding to this config.
        config : dict
            The dictionary of the config itself.
        """

        # Construct all permutations over k and M, N, and eta:
        all_perms = list(product(
            self.args.M, self.args.N, config['k_units_pi'], self.args.eta)
        )

        # Get the omega grid
        wgrids = self._compute_overall_grid(config)

        # Save the config file itself
        with open(f"{base}/config.yaml", 'w') as outfile:
            yaml.dump(
                config, outfile, default_flow_style=False, allow_unicode=False
            )

        # One submit script per config directory
        utils.run_command(f"cp {slurm_path} {base}")

        global_dict = dict()
        ex = 'sbatch'  # method for running the program
        for cc, (mm, nn, kk, ee) in enumerate(all_perms):
            global_dict[cc] = {
                'M': mm, 'N': nn, 'lambda': config['lambda'], 'k_units_pi': kk,
                'eta': ee
            }

            # The number of wgrids is determined by the distribute flag in the
            # command line arguments
            for grid_counter, grid in enumerate(wgrids):
                final_path = f"{base}/{cc:03}/{grid_counter:03}"

                # Make the directory for this grid
                os.makedirs(final_path, exist_ok=True)

                # Save the grid
                grid_path = f"{final_path}/grid.txt"
                np.savetxt(grid_path, grid)

                # State directory for containing the current progress of the
                # computation
                os.makedirs(f"{final_path}/state")

                # Save the input parameters in each of the running directories
                input_params = InputParameters(
                    M=mm, N=nn, eta=ee, t=config['t'], Omega=config['Omega'],
                    lambd=config['lambda'], model=config['model'],
                    config_filter=config['config_filter'], k=kk
                )
                input_params.save_config(f"{final_path}/config.yaml")

                # Save the script itself, which should be ran from the
                # working directory
                with open(f"{final_path}/script.sh", 'w') as f:
                    f.write(f"mv {base}/submit.sbatch .\n")
                    f.write(f'{ex} submit.sbatch {final_path} "$@"\n')
                    f.write(f"mv submit.sbatch {base}\n")

        with open(f"{base}/mapping.yaml", 'w') as outfile:
            yaml.dump(global_dict, outfile, default_flow_style=False)

    def _ready_package(self, package_path_cache, loaded_configs, slurm_path):
        """Prepares a single package for running."""

        for ii, config in enumerate(loaded_configs):
            dlog.info(f"Readying config {ii:03}")
            base = f"{package_path_cache}/{ii:03}"
            os.makedirs(base)

            # Parse each configuration's parameters specifically
            if config['linspacek']:
                if len(config['k_units_pi']) != 3:
                    msg = \
                        f"Invalid k linspace parameters {config['k_units_pi']}"
                    dlog.critical(msg)
                    raise RuntimeError(msg)
                config['k_units_pi'] = np.linspace(
                    *config['k_units_pi'], endpoint=True
                ).tolist()

            self._ready_config(base, config, slurm_path)

    def prime(self):
        """Prepares all computations staged in packages (or specifically the
        user-selected ones) for execution by the job controller by creating
        all appropriate directories."""

        # Detect the staged packages
        package = self.args.package
        if package is None:
            staged_packages = self.all_packages

        else:
            staged_packages = [
                p for p in self.all_packages if f"{package:03}" in p
            ]

            # Should only find one package in this case
            if len(staged_packages) != 1:
                msg = "Detected more than one staged package for " \
                    f"package {package}"
                dlog.critical(msg)
                raise RuntimeError(msg)

        # With the list of staged packages now inplace, we run every config
        # in those packages
        for ii, pack_path_load in enumerate(staged_packages):

            # Get the package information
            full_package_name = os.path.basename(pack_path_load)
            pack_name = full_package_name.split("_")[1]
            pack_num = full_package_name.split("_")[0]
            dlog.info(f"Readying package #{pack_num}: {pack_name}")

            # Make the package directory
            package_path_cache = \
                f"{self.cache_dir}/{self.current_index:03}_{pack_name}"

            # Create the target directory in the cache
            os.makedirs(package_path_cache, exist_ok=False)

            # Load in all files
            all_files = utils.listdir_fullpath(pack_path_load)
            staged_configs = [s for s in all_files if "config.yaml" in s]
            configs = [yaml.safe_load(open(f)) for f in staged_configs]
            slurm_path = f"{pack_path_load}/submit.sbatch"

            # Ready that specific package
            self._ready_package(package_path_cache, configs, slurm_path)
            self.current_index += 1
