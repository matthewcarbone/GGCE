#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


from datetime import datetime
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

    def __init__(self, cl_args):
        self.cl_args = dict(vars(cl_args))
        self.loaded_config = yaml.safe_load(
            open(self.cl_args['loaded_config_path'])
        )

    def get_val(self, key):
        """Checks if a key occurs in the command line args, if not, checks
        the config, else if no key exists, returns None."""

        val = self.cl_args.get(key)
        if val is not None:
            return val
        val = self.loaded_config.get(key)
        return val

    def _get_requeue_Cori(self):
        """Get's the requeue lines for the Cori supercomputer."""

        t_min = self.get_val("t_min")
        if t_min is None:
            msg = "t_min is required for requeue job"
            dlog.critical(msg)
            raise RuntimeError(msg)

        t_total = self.get_val("t_total")
        if t_total is None:
            msg = "t_total is required for requeue job"
            dlog.critical(msg)
            raise RuntimeError(msg)

        return [
            f"#SBATCH --time-min={t_min}",
            f"#SBATCH --comment={t_total}",
            f"#SBATCH --signal=B:USR1@60",   # catch signal 60 s from kill
            "#SBATCH --requeue",
            "#SBATCH --open-mode=append\n",
            "ckpt_command=\n",
            ". /usr/common/software/variable-time-job/setup.sh",
            "requeue_job func_trap USR1",
            "#\n",
        ]

    def get_requeue_lines(self):
        """Get's the specific arguments necessary for a requeue job. This is
        a bit different since it requires specific arguments and depends on
        the cluster.
        """

        cluster = self.loaded_config.get("cluster")
        if cluster is None:
            msg = "Cluster required for requeue job"
            dlog.critical(msg)
            raise RuntimeError(msg)

        if cluster == "Cori":
            return self._get_requeue_Cori()
        else:
            msg = f"Unsupported cluster {cluster} for requeue"
            dlog.critical(msg)
            raise RuntimeError(msg)

    def get_other_lines(self):
        """Lines corresponding to other quantities, like the number of threads
        or the modules."""

        lines = []

        # Threads -------------------------------------------------------------
        threads_per_task = self.get_val("threads_per_task")
        if threads_per_task is not None:
            lines.append(f"export OMP_NUM_THREADS={threads_per_task}")
        else:
            lines.append(f"export OMP_NUM_THREADS=1")
            dlog.warning("Threads/process not set, defaulting to 1")

        # Modules -------------------------------------------------------------
        # Only check the config for this
        modules = self.loaded_config.get("modules")
        if modules is not None:
            for mod in modules:
                lines.append(f"module load {mod}")

        return lines

    def get_standard_lines(self, stream_name=None):
        """Iterates through the various possible lines that the SLURM script
        can have (these are hard-coded, since it appears this is one of those
        times where over-engineering the solution is a real possibility),
        and returns a list of those lines, exclusing the newline flags. Note
        this method only get's the standard lines such as the partition,
        output/error stream, job name, etc.

        The keys this method will check are
            'partition', 'job_name', 'job_data_directory', 'N_nodes',
        'memory_per_node', 'tasks_per_node', 'constraint', 'QOS', 'email',
        't_max'
        """

        SBATCH_lines = []

        # Partition -----------------------------------------------------------
        partition = self.get_val("partition")
        if partition == 'no_partition':
            pass
        elif partition is not None:
            SBATCH_lines.append(f"#SBATCH -p {partition}")
        else:
            dlog.warning("Partition unspecified and not `no_partition`")

        # Constraint ----------------------------------------------------------
        constraint = self.get_val("constraint")
        if constraint == 'no_constraint':
            pass
        elif constraint is not None:
            SBATCH_lines.append(f"#SBATCH --constraint={constraint}")
        else:
            dlog.warning("Constraint unspecified and not `no_constraint`")

        # Constraint ----------------------------------------------------------
        qos = self.get_val("qos")
        if qos == 'no_qos':
            pass
        elif qos is not None:
            SBATCH_lines.append(f"#SBATCH -q {qos}")
        else:
            dlog.warning("QOS unspecified and not `no_qos`")

        # Max runtime ---------------------------------------------------------
        t_max = self.get_val("t_max")
        if t_max == 'no_t_max':
            pass
        elif t_max is not None:
            assert isinstance(t_max, str)
            SBATCH_lines.append(f"#SBATCH --time={t_max}")
        else:
            dlog.warning("Max time limit unspecified and not `no_t_max`")

        # Mail ----------------------------------------------------------------
        email = self.get_val("email")
        if email is not None:
            SBATCH_lines.append(f"#SBATCH --mail-user={email}")
            SBATCH_lines.append(f"#SBATCH --mail-type=ALL")

        # Job name ------------------------------------------------------------
        job_name = self.get_val("job_name")
        if job_name is not None:
            SBATCH_lines.append(f"#SBATCH -J {job_name}")
        else:
            SBATCH_lines.append(f"#SBATCH -J GGCE")

        # Out/err stream directories ------------------------------------------
        job_data_directory = self.get_val("job_data_directory")
        if job_data_directory is not None:
            base_directory = job_data_directory
        else:
            base_directory = "job_data"
        if stream_name is None:
            SBATCH_lines.append(
                f"#SBATCH --output={base_directory}/GGCE_%A.out"
            )
            SBATCH_lines.append(
                f"#SBATCH --error={base_directory}/GGCE_%A.err"
            )
        else:
            SBATCH_lines.append(
                f"#SBATCH --output={base_directory}/{stream_name}_%A.out"
            )
            SBATCH_lines.append(
                f"#SBATCH --error={base_directory}/{stream_name}_%A.err"
            )

        # Nodes ---------------------------------------------------------------
        N_nodes = self.get_val("N_nodes")
        if N_nodes is not None:
            assert isinstance(N_nodes, int)
            SBATCH_lines.append(f"#SBATCH -N {N_nodes}")
        else:
            SBATCH_lines.append(f"#SBATCH -N 1")

        # Memory per node -----------------------------------------------------
        mem_per_node = self.get_val("mem_per_node")
        if mem_per_node is not None:
            assert isinstance(mem_per_node, str)
            SBATCH_lines.append(f"#SBATCH --mem={mem_per_node}")

        # MPI tasks/node ------------------------------------------------------
        tasks_per_node = self.get_val("tasks_per_node")
        if tasks_per_node is not None:
            assert isinstance(tasks_per_node, int)
            SBATCH_lines.append(f"#SBATCH --tasks-per-node={tasks_per_node}")
        else:
            SBATCH_lines.append(f"#SBATCH --tasks-per-node=1")
            dlog.warning("Tasks per node is not set, defaulting to 1")

        return SBATCH_lines

    def write(self, target, stream_name=None):
        """Takes command line arguments, initializes the configuration and
        writes the new SLURM script to disk. We only want to override the
        default values in the config if the command line values are not None.
        This method parses these two dictionaries accordingly."""

        standard_lines = self.get_standard_lines(stream_name=stream_name)
        requeue_lines = self.get_requeue_lines() if self.cl_args['requeue'] \
            else []
        other_lines = self.get_other_lines()

        # The last line is always the same unless requeue
        if self.cl_args['requeue']:
            last_line = 'srun python3 ._submit.py "$@" &\nwait'
        else:
            last_line = 'srun python3 ._submit.py "$@"'

        with open(target, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for line in standard_lines:
                f.write(f"{line}\n")
            f.write("\n")
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

    def __init__(self, cl_args):

        now = datetime.now()
        self.dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")

        # The directories important for everything we do here.
        self.cache_dir = utils.get_cache_dir()
        self.package_dir = utils.get_package_dir()
        self.cl_args = cl_args

        # Select the user-specified package to run
        package_name = self.cl_args.package
        package_dir = self.cl_args.package_dir
        self.staged_package_path = os.path.join(package_dir, package_name)
        if not os.path.isdir(self.staged_package_path):
            msg = f"Package path {self.staged_package_path} does not exist"
            dlog.critical(msg)
            raise RuntimeError(msg)

        msg = "Selected Package " + utils.bold(f"{self.staged_package_path}")
        dlog.info(msg)

    def _get_all_packages(self):
        """Gets all non-template packages from the packages directory.
        Returns a list of the full paths."""

        all_packages = utils.listdir_fullpath(self.package_dir)
        all_packages = [
            p for p in all_packages
            if ("TEMPLATE" not in p and ".gitignore" not in p)
        ]
        dlog.debug(
            f"Found a total of {len(all_packages)} packages in "
            f"{self.package_dir}"
        )
        return all_packages

    def _setup_cache_target(self):
        """Creates the necessary directories in the cache."""

        basename = os.path.basename(self.staged_package_path)
        basename = f"{basename}_{self.dt_string}"
        if self.cl_args.info is not None:
            basename = f"{basename}_{self.cl_args.info}"
        pack_name = os.path.join(self.cache_dir, basename)

        # Create the target directory in the cache
        dlog.debug(f"Making package directory {pack_name}")
        os.makedirs(pack_name, exist_ok=False)
        bold_str = utils.bold(f"-P {basename}")
        dlog.info(f"Submit with {bold_str}")
        return pack_name

    def _get_config_files(self):
        """Returns a list of all the staged config files in this package,
        sorted. Note that config files must contain the substring
        `config.yaml`."""

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
        n_kpts = len(N_M_eta_k[3])
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
        total *= n_kpts
        dlog.info(f"Total {total} k x w-points primed for computation")

    def scaffold(self):
        """Primes the computation by constructing all permutations of jobs
        necessary to be run, and saving this as a single pickle file to disk.
        This file is then read by the RANK=0 MPI process and distributed to
        the workers during execution."""

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

        basename = self.cl_args.package
        package = os.path.join(self.cache_dir, basename)
        if not os.path.isdir(package):
            msg = f"Cached package path {package} does not exist"
            dlog.critical(msg)
            raise RuntimeError(msg)

        slurm_writer = SlurmWriter(self.cl_args)

        dlog.debug(f"Package path {package}")
        submit_script = os.path.join(package, "submit.sbatch")

        # Write the slurm script regardless of --bash, why not
        slurm_writer.write(submit_script, stream_name=basename)

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

            local_threads = self.cl_args.threads_per_task
            if local_threads is None:
                local_threads = 1
                dlog.warning("Local threads not set, defaulting to 1")

            exe = f"mpiexec -np {local_procs} python3 ._submit.py " \
                f"{package} {debug} {dryrun}"
            fname = f"submit_{basename}.sh"
            with open(fname, 'w') as f:
                f.write(f"export OMP_NUM_THREADS={local_threads}\n")
                f.write(exe)
            dlog.info(f"Run {fname} to execute trials")

        # Else we go through the protocol of submitting a job via SLURM
        else:
            utils.run_command(f"mv {submit_script} .")
            args = f"{package} {debug} {dryrun}"
            out = utils.run_command(f"sbatch submit.sbatch {args}")
            if out == 0:
                dlog.info(f"{basename} submit - success")
            else:
                dlog.error(f"{basename} submit - failure (err code {out})")
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
