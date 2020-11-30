#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import copy
import numpy as np
import os
import pickle
import yaml

from ggce.engine.structures import InputParameters
from ggce.engine import system
from ggce.utils import utils
from ggce.utils.logger import default_logger as dlog


class SlurmWriter:
    """Writes a SLURM script from scratch.

    Parameters
    ----------
    target_dir : str
        The full path location to the directory that will contain the SLURM
        submit script.
    default_config : dict
        A dictionary containing the default configurations for the SLURM
        script. These will be overridden by command line arguments.
    """

    # Determines the mapping between the config keys and the flag SLURM needs
    KEYMAP = {
        'partition': lambda s: f"#SBATCH -p {s}",
        'jobname': lambda s: f"#SBATCH -J {s}",
        'output': lambda s: f"#SBATCH --output={s}",
        'error': lambda s: f"#SBATCH --error={s}",
        'memory': lambda s: f"#SBATCH --mem={s}",
        'nodes': lambda s: f"#SBATCH -N {s}",
        'tasks_per_node': lambda ii: f"#SBATCH --tasks-per-node={ii}",
        'constraint': lambda s: f"#SBATCH --constraint={s}",
        'time': lambda s: f"#SBATCH --time={s}",
        'account': lambda s: f"#SBATCH -A {s}",
        'gres': lambda ii: f"#SBATCH --gres=gpu:{ii}",
        'queue': lambda s: f"#SBATCH -q {s}",
        'email': lambda email_address: f"#SBATCH --mail-user={email_address}",
        'mail_type': lambda s: f"#SBATCH --mail-type={s}",
        'time_min': lambda s: f"#SBATCH --time-min={s}"
    }

    # Maps basically everything else to the proper format
    OTHERMAP = {
        'threads': lambda ii=None:
            f"export OMP_NUM_THREADS={ii}" if ii is not None else
            "export OMP_NUM_THREADS=1",
        'disable_kmp_affinity': lambda b:
            f"export KMP_AFFINITY=disabled" if b else None,
        'omp_places': lambda s: f"export OMP_PLACES={s}",
        'omp_proc_bind': lambda s: f"export OMP_PROC_BIND={s}",
        'modules': lambda list_of_modules=None:
            "\n".join([f"module load {m}" for m in list_of_modules])
            if list_of_modules is not None else None
    }

    def __init__(self, cl_args):
        self.cl_args = dict(vars(cl_args))
        self.loaded_config = yaml.safe_load(
            open(self.cl_args['loaded_config_path'])
        )

    @staticmethod
    def _check_key(key, value):

        if value is not None:

            mapped = SlurmWriter.KEYMAP.get(key)
            if mapped is not None:
                mapped = mapped(value)
                if mapped is not None:
                    return (0, mapped)

            mapped = SlurmWriter.OTHERMAP.get(key)
            if mapped is not None:
                mapped = mapped(value)
                if mapped is not None:
                    return (1, mapped)

        return (None, None)

    @staticmethod
    def requeue_lines(target, total_time, checkpoint_time=10):
        """Gets a specific part of a SLURM script for the KNL flex queue."""

        return [
            f"#SBATCH --comment={total_time}",
            f"#SBATCH --signal=B:USR1@{checkpoint_time}",
            "#SBATCH --requeue",
            "#SBATCH --open-mode=append\n",
            "ckpt_command=\n",
            ". /usr/common/software/variable-time-job/setup.sh",
            "requeue_job func_trap USR1",
            "#\n",
        ]

    def write(self, target):
        """Takes command line arguments, initializes the configuration and
        writes the new SLURM script to disk. We only want to override the
        default values in the config if the command line values are not None.
        This method parses these two dictionaries accordingly."""

        lines = []
        other_lines = []

        master_dict = copy.copy(self.loaded_config)
        keymap_keys = list(SlurmWriter.KEYMAP.keys())
        othermap_keys = list(SlurmWriter.OTHERMAP.keys())
        d_keys = list(self.loaded_config.keys())

        # Iterate over the CL args
        for key, value in self.cl_args.items():

            # If we find a key that matches a key in the mappings provided in
            # this class
            if key in keymap_keys or key in othermap_keys:

                # And, if that key matches a key in the default parameters
                if key in d_keys:

                    # Then continue, since this case will be caught in the next
                    # loop
                    continue

                # Otherwise, if they key does not match, we add that key to
                # the list
                master_dict[key] = value

        # Iterate over the full list, which is the union of the keys in the
        # CL args which match the mappings provided in this class, and the
        # default args found in the config
        for key, value in master_dict.items():

            # Define a temporary variable, tmp, which stores the value
            # corresponding to the key. This could be None/null, or it could be
            # some value
            tmp = self.cl_args.get(key)

            # If the CL arg doesn't exist, or defaults to None, use the
            # key-value pair found in the default config
            value = value if tmp is None else tmp

            # Get the actual line to write to the SLURM script. It is possible
            # based on the arguments that there is nothing to write, and this
            # will also handle those cases. This also differentiates between
            # the SLURM parmaeters and the "others", which are not SLURM
            # parameters, and are written to the submit script differently.
            (loc, val) = SlurmWriter._check_key(key, value)
            if loc == 0:
                lines.append(val)
            elif loc == 1:
                other_lines.append(val)

        # The last line is always the same
        last_line = 'srun python3 ._submit.py "$@"'

        with open(target, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for line in lines:
                f.write(f"{line}\n")
            if self.cl_args['requeue']:
                last_line = f"{last_line} &\nwait"
                assert 'total_time' in list(self.loaded_config.keys())
                assert 'time' in list(self.loaded_config.keys())
                assert 'time_min' in list(self.loaded_config.keys())
                requeue_lines = SlurmWriter.requeue_lines(
                    target, self.loaded_config['total_time']
                )
                for line in requeue_lines:
                    f.write(f"{line}\n")
            f.write("\n")
            for line in other_lines:
                f.write(f"{line}\n")
            f.write(f"\n{last_line}")


class Prime:
    """Prepares the computation for submission by evaluating all jobs to be
    run, and saving them to a working directory.

    TODO: detailed docstring
    """

    def _get_all_packages(self):
        """Gets all non-template packages from the packages directory.
        Returns a list of the full paths."""

        all_packages = utils.listdir_fullpath(self.package_dir)
        all_packages = [
            p for p in all_packages
            if ("TEMPLATE" not in p and ".gitignore" not in p)
        ]
        all_packages.sort()
        dlog.debug(
            f"Found a total of {len(all_packages)} packages in "
            f"{self.package_dir}"
        )
        return all_packages

    @staticmethod
    def _get_user_selected_package(all_packages, package_no):
        """Get's the user-selected package by package number."""

        user_selected = [
            p for p in all_packages
            if f"{package_no:03}" in os.path.basename(p)
        ]

        # Throw an error if there are more than one packages corresponding
        # to the same user-specified package number.
        if len(user_selected) != 1:
            msg = "Found >1 packages corresponding to the same " \
                f"user-specified value {package_no} ({package_no:03}) "
            dlog.critical(msg)
            msg = "Unable to continue - resolve ambiguity and re-run"
            dlog.critical(msg)
            raise RuntimeError(msg)

        return user_selected[0]

    def _get_current_cache_index(self):
        """Get the basename of the last file and add one to the index. This
        marks the index to resume creating directories at."""

        # List all of the directories already existing in the cache (with full
        # paths). Ignore system/hidden files with != ".".
        existing = utils.listdir_fullpath(self.cache_dir)
        existing = [e for e in existing if os.path.basename(e)[0] != "."]
        existing.sort()

        # Get the basename of the last file and add one to the index. This
        # marks the index to resume creating directories at.
        if len(existing) > 0:
            dlog.debug(f"Found {len(existing)} dirs already in the cache")
            dlog.debug(f"These caches are {existing}")
            basename = os.path.basename(existing[-1])
            current_index = int(basename.split("_")[0]) + 1
            sub_flag = utils.bold(f"-P {current_index}")
            dlog.info(
                f"Found {current_index} existing: after scaffolding, "
                f"to submit, use flag {sub_flag}"
            )
        else:
            current_index = 0
            sub_flag = utils.bold(f"-P {current_index}")
            dlog.info(f"To submit, use flag {sub_flag}")

        if current_index > 999:
            dlog.error("Current index > 999, expect unexpected behavior")

        return current_index

    def __init__(self, cl_args):

        # The directories important for everything we do here.
        self.cache_dir = utils.get_cache_dir()
        self.package_dir = utils.get_package_dir()
        self.cl_args = cl_args

        # Select the user-specified package to run, and find the corresponding
        # name of the package
        package_no = cl_args.package
        all_packages = self._get_all_packages()
        self.staged_package_path = \
            Prime._get_user_selected_package(all_packages, package_no)
        msg = "Selected Package " + utils.bold(f"{package_no:03}") \
            + f" ({self.staged_package_path})"
        dlog.info(msg)

        # Get the current index of the cache
        self.current_index = self._get_current_cache_index()

    def _setup_cache_target(self):
        """Creates the necessary directories in the cache."""

        full_package_name = os.path.basename(self.staged_package_path)
        pack_name = full_package_name.split("_")[1]
        pack_num = full_package_name.split("_")[0]
        dlog.debug(f"Package num/name: {pack_num}/{pack_name}")

        # Make the package directory
        package_path_cache = \
            f"{self.cache_dir}/{self.current_index:03}_{pack_name}"

        if self.cl_args.info is not None:
            package_path_cache = f"{package_path_cache}_{self.cl_args.info}"

        # Create the target directory in the cache
        dlog.debug(f"Making package directory {package_path_cache}")
        os.makedirs(package_path_cache, exist_ok=False)
        return package_path_cache

    def _get_config_files(self):
        """Returns a list of all the staged config files in this package,
        sorted."""

        # Load in all files
        all_files = utils.listdir_fullpath(self.staged_package_path)

        if self.cl_args.c_to_run is not None:
            c_to_run = [f"{xx:02}" for xx in self.cl_args.c_to_run]

        tmp_staged_configs = [s for s in all_files if "config.yaml" in s]
        tmp_staged_configs.sort()
        configs = dict()
        for jj, f in enumerate(tmp_staged_configs):
            fname = f.split("/")[-1]

            # Skip configs not specified
            if self.cl_args.c_to_run is not None:
                fn = int(fname[:2])

                # We only use the first two numbers to index the config
                if f"{fn:02}" not in c_to_run:
                    dlog.debug(f"Skipping config {fname}")
                    continue

            configs[fname] = yaml.safe_load(open(f))
            dlog.debug(f"Preparing config {fname}")

        return configs

    @staticmethod
    def _get_M_N_eta_k_mapping(M_N_eta_k):
        """Returns a nested dictionary containing the mappings between the
        following variables: config -> M -> N -> eta."""

        return {
            'M': {M: cc for cc, M in enumerate(M_N_eta_k[0])},
            'N': {N: cc for cc, N in enumerate(M_N_eta_k[1])},
            'eta': {eta: cc for cc, eta in enumerate(M_N_eta_k[2])},
            'k_units_pi': {k: cc for cc, k in enumerate(M_N_eta_k[3])}
        }

    def _ready_configs(self, configs, package_cache_path):
        """Uses the list of configs passed and the command line information
        to produce the computation-ready input file information.
            The idea is to create a mapping:
            omega + config -> directory location
        """

        if self.cl_args.linspacek:
            if len(self.cl_args.k_units_pi) != 3:
                msg = "With --linspacek specified, -k requires 3 arguments: " \
                    "k0, kf and the number of k-points"
                dlog.critical(msg)
                raise RuntimeError(msg)

        N_M_eta_k = [
            self.cl_args.M, self.cl_args.N, self.cl_args.eta,
            list(np.linspace(*self.cl_args.k_units_pi, endpoint=True))
            if self.cl_args.linspacek else self.cl_args.k_units_pi
        ]
        M_N_eta_k_mapping = Prime._get_M_N_eta_k_mapping(N_M_eta_k)

        # Maps the config index to the object itself
        config_mapping = dict()

        # Master list of every job to execute. Maps the config index to a
        # list of frequency gridpoints
        master_mapping = dict()

        cc = 0
        total = 0
        for config_name, config in configs.items():

            if 'linspacek' in list(config.keys()):
                dlog.warning("k-entries in config is deprecated")

            # Get the frequency grid
            grid = list(np.sort(np.concatenate([
                np.linspace(*c, endpoint=True)
                for c in config['linspace_params']
            ])))
            total += len(grid)

            # Each config gets its own directory in the package_cache_path
            config_mapping[cc] = config
            master_mapping[cc] = grid
            dlog.debug(f"Prepared config {config_name} ({cc})")

            cc += 1

        pickle_dict = (
            master_mapping, config_mapping, M_N_eta_k_mapping,
            package_cache_path
        )
        pickle_path = os.path.join(package_cache_path, "protocol.pkl")
        pickle.dump(pickle_dict, open(pickle_path, 'wb'), protocol=4)
        dlog.info(f"Total {total} w-points primed for computation")

    def scaffold(self):
        """Primes the computation by constructing all permutations of jobs
        necessary to be run, and saving this as a single pickle file to disk.
        This file is then read by the RANK=0 MPI process and distributed to
        the workers during execution."""

        package_str = f"{self.cl_args.package:03}"
        dlog.info(
            f"Scaffolding package {utils.bold(package_str)} as cache ID "
            + utils.bold(f"{self.current_index:03}")
        )
        package_cache_path = self._setup_cache_target()
        configs = self._get_config_files()
        self._ready_configs(configs, package_cache_path)


class Submitter:
    """TODO: detailed docstring"""

    def __init__(self, cl_args):

        # The directories important for everything we do here.
        self.cache_dir = utils.get_cache_dir()
        self.package_dir = utils.get_package_dir()
        self.cl_args = cl_args

    def submit_mpi(self):
        """Submits the packages as specified by the cl_args."""

        all_cache = utils.listdir_fullpath(self.cache_dir)
        all_cache.sort()

        packs_to_run = None
        if self.cl_args.package is not None:
            packs_to_run = [f"{p:03}" for p in self.cl_args.package]

        slurm_writer = SlurmWriter(self.cl_args)
        for ii, package in enumerate(all_cache):
            if self.cl_args.package is not None:
                if f"{ii:03}" not in packs_to_run:
                    continue
            dlog.debug(f"Package path {package}")
            submit_script = f"{package}/submit.sbatch"

            # Write the slurm script regardless of --bash, why not
            slurm_writer.write(submit_script)

            dryrun = int(self.cl_args.dryrun)
            debug = int(self.cl_args.debug)

            # If bash is true, then we save a local run script, which the
            # user can run separately using bash.
            if self.cl_args.bash:
                # Since the local machine is basically a node, we use the
                # tasks/node CL argument to determine how many processes to
                # spawn on the local machine
                local_procs = self.cl_args.tasks_per_node
                if local_procs is None:
                    local_procs = 1
                    dlog.warning("Local procs not set, defaulting to 1")

                exe = f"mpiexec -np {local_procs} python3 ._submit.py " \
                    f"{package} {debug} {dryrun}"
                fname = f"submit_{ii:03}.sh"
                with open(fname, 'w') as f:
                    f.write(f"export OMP_NUM_THREADS={self.cl_args.threads}\n")
                    f.write(exe)
                dlog.info(f"Run {fname} to execute trials")

            # Else we go through the protocol of submitting a job via SLURM
            else:
                utils.run_command(f"mv {submit_script} .")
                args = f"{package} {debug} {dryrun}"
                out = utils.run_command(f"sbatch submit.sbatch {args}")
                if out == 0:
                    dlog.info(f"Cache {ii:03} submit - success")
                else:
                    dlog.error(
                        f"Cache {ii:03} submit - failure (err code {out})"
                    )
                utils.run_command(f"mv submit.sbatch {package}")


class Auxiliary:
    """A class containing debugging and other methods for studying the
    structure of the produced equations."""

    @staticmethod
    def analyze_XN(M, N, model='H'):
        """For any given M and N, there is a hierarchy of equations generated
        in the number of bosons. At every configuration with n bosons, it
        couples to all legally-accessible configurations with n pm 1 bosons.
        This method analyzes the precise number of equations at each
        level of the hierarchy, for specified M and max N. Note that this
        result also depends on the model, which is Holstein by default."""

        with utils.DisableLogger():
            input_params = InputParameters(
                M=M, N=N, eta=1.0, t=1.0, Omega=1.0, lambd=1.0, model=model,
                config_filter='no_filter'
            )
            input_params.init_terms()
            sy = system.System(input_params)
            sy.initialize_generalized_equations()
            sy.initialize_equations()

        # These are the equations of the closure.
        dat = [(key, len(value)) for key, value in sy.equations.items()]
        dat.sort(key=lambda x: x[0])
        return [d[0] for d in dat], [d[1] for d in dat]
