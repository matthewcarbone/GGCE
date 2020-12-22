#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


from datetime import datetime
from math import floor
import os
import yaml

from ggce.engine.structures import InputParameters
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
            "#SBATCH --signal=B:USR1@60",   # catch signal 60 s from kill
            "#SBATCH --requeue",
            "#SBATCH --open-mode=append\n",
            "ckpt_command=\n",
            ". /usr/common/software/variable-time-job/setup.sh",
            "requeue_job func_trap USR1",
            "#\n",
        ]

    def _get_requeue_Habanero(self):
        """Get's the requeue lines for the Habanero cluster."""

        requeue_fn = """\n
requeue_job()
{
    echo "requeue n.o. $SLURM_RESTART_COUNT"
    date
    scontrol requeue $SLURM_JOBID
}\n
        """

        return [
            "#SBATCH --signal=B:USR1@60",  # catch sig term 60 s from end
            "#SBATCH --requeue",  # Enable requeuing
            requeue_fn,
            "trap 'requeue_job' USR1"
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

        if cluster == "Cori" or cluster == "cori":
            return self._get_requeue_Cori()

        # Using rr to test the requeue script
        elif cluster == "Habanero" or cluster == "habanero" or cluster == "rr":
            return self._get_requeue_Habanero()
        else:
            msg = f"Unsupported cluster {cluster} for requeue"
            dlog.critical(msg)
            raise RuntimeError(msg)

    def get_other_lines(self, phys_cores_per_task):
        """Lines corresponding to other quantities, like the number of threads
        or the modules."""

        lines = []

        # Modules -------------------------------------------------------------
        # Only check the config for this
        other_lines = self.loaded_config.get("other_lines")
        if other_lines is not None:
            for line in other_lines:
                lines.append(f"{line}")

        # Threads -------------------------------------------------------------
        max_threads = self.loaded_config.get("max_threads")
        if max_threads is None:
            threads = phys_cores_per_task
        else:
            threads = min(phys_cores_per_task, max_threads)
        lines.append(f"export OMP_NUM_THREADS={threads}")

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

        # Account -------------------------------------------------------------
        account = self.get_val("account")
        if account == 'no_account':
            pass
        elif account is not None:
            SBATCH_lines.append(f"#SBATCH --account={account}")
        else:
            dlog.warning("Account unspecified and not `no_account`")

        # QOS -----------------------------------------------------------------
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

        # Exclusive flag ------------------------------------------------------
        exclusive = self.get_val("exclusive")
        if exclusive is not None:
            if exclusive:
                SBATCH_lines.append(f"#SBATCH --exclusive")

        # Job name ------------------------------------------------------------
        job_name = self.get_val("job_name")
        if job_name is not None:
            SBATCH_lines.append(f"#SBATCH -J {job_name}")
        else:
            SBATCH_lines.append(f"#SBATCH -J GGCE")

        # Out/err stream directories ------------------------------------------
        base_directory = utils.JOB_DATA_PATH
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
        # The user can specify some total number of nodes N and the number of
        # tasks per node. The total number of tasks is obviously
        # N * n_tasks_per_node
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
        cores_per_node = self.get_val("cores_per_node")
        if cores_per_node is None:
            msg = "Set the cores_per_node value in your SLURM config"
            dlog.critical(msg)
            raise RuntimeError(msg)
        hyperthreads_per_core = self.get_val("hyperthreads_per_core")
        if hyperthreads_per_core is None:
            msg = "Set the hyperthreads_per_core value in your SLURM config"
            dlog.critical(msg)
            raise RuntimeError(msg)

        tasks_per_node = self.get_val("tasks_per_node")
        if tasks_per_node is None:
            msg = "Set the tasks_per_node value in your SLURM config"
            dlog.critical(msg)
            raise RuntimeError(msg)
        assert isinstance(tasks_per_node, int)
        SBATCH_lines.append(f"#SBATCH --tasks-per-node={tasks_per_node}")
        phys_cores_per_task = int(floor(cores_per_node/tasks_per_node))
        c = phys_cores_per_task * hyperthreads_per_core
        SBATCH_lines.append(f"#SBATCH -c {c}")

        return SBATCH_lines, phys_cores_per_task

    def write(self, target, stream_name=None):
        """Takes command line arguments, initializes the configuration and
        writes the new SLURM script to disk. We only want to override the
        default values in the config if the command line values are not None.
        This method parses these two dictionaries accordingly."""

        standard_lines, phys_cores_per_task \
            = self.get_standard_lines(stream_name=stream_name)
        requeue_lines = self.get_requeue_lines() if self.cl_args['requeue'] \
            else []
        other_lines = self.get_other_lines(phys_cores_per_task)
        bind_cores = self.get_val("bind_cores")

        cluster = self.loaded_config.get("cluster")

        # The last line is always the same unless requeue
        if cluster == "Cori":
            bind_str = " --cpu-bind=cores" if bind_cores else ""
            if self.cl_args['requeue']:
                last_line = f'srun{bind_str} python3 ._submit.py "$@" &\nwait'
            else:
                last_line = f'srun{bind_str} python3 ._submit.py "$@"'
        elif cluster == "rr":
            last_line = f'mpiexec python3 ._submit.py "$@"'
        elif cluster == "habanero":
            if self.cl_args['requeue']:
                last_line = f'mpiexec python3 ._submit.py "$@" &\nwait'
            else:
                last_line = f'mpiexec python3 ._submit.py "$@"'
        else:
            msg = f"Unknown cluster {cluster}"
            dlog.critical(msg)
            raise RuntimeError(msg)

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


class BaseOverlord:
    def __init__(self, cl_args):
        self.cache_dir = utils.get_cache_dir()
        self.package_dir = utils.get_package_dir()
        self.cl_args = cl_args
        self.queue_path = utils.LIFO_QUEUE_PATH


class Prime(BaseOverlord):
    """Prepares the computation for submission by evaluating all jobs to be
    run, and saving them to a working directory.

    Attributes
    ----------
    dt_string : str
        Datetime set at the time the class is initialized.
    cache_dir : str
        Path to the location for saving the results. Checks the environment
        variable GGCE_CACHE_DIR; defaults to a local directory "cache".
    package_dir : str
        Path to the location of the packages. Checks the environment variable
        GGCE_PACKAGES_DIR; defaults to local directory "packages".
    cl_args : dict
        Dictionary representation of the command line arguments.
    staged_package_path : str
        The specific path to the directory for storing results for this
        particular trial.

    Parameters
    ----------
    cl_args
        Command line arguments Namespace directly from argparse.
    """

    def __init__(self, cl_args):
        super().__init__(cl_args)

        now = datetime.now()
        self.dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")

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
        return pack_name, basename

    def _get_config_files(self):
        """Returns a list of all the staged config files in this package,
        sorted. Note that config files must contain the substring
        `config.yaml`."""

        # Load in all files
        all_files = utils.listdir_fullpath(self.staged_package_path)

        tmp_staged_configs = [s for s in all_files if ".yaml" in s]
        configs = dict()

        if len(tmp_staged_configs) < 1:
            dlog.error("Len of staged configs is less than 1")

        for jj, f in enumerate(tmp_staged_configs):
            fname = f.split("/")[-1]

            # Skip configs not specified
            if self.cl_args.c_to_run is not None:
                if fname not in self.cl_args.c_to_run:
                    dlog.debug(f"Skipping config {fname}")
                    continue

            configs[fname] = yaml.safe_load(open(f))

        return configs

    def _ready_package(self, configs, package_cache_path):
        """Uses the list of configs passed and the command line information
        to produce the computation-ready input file information."""

        # Iterate over the loaded config dictionaries
        cl_args = dict(vars(self.cl_args))
        package_configs_directory = os.path.join(package_cache_path, "configs")
        os.makedirs(package_configs_directory, exist_ok=False)

        for config_name, config_dict in configs.items():

            # For each config dictionary, initialize the InputParameters
            # object, and parse that, iterating through it to produce all
            # M/N permutations
            upper_inp = InputParameters(config_dict, cl_args)

            for sub_cc, inp in enumerate(upper_inp):
                name = f"{sub_cc:03}_{config_name}"
                t = os.path.join(package_configs_directory, name)
                inp.save(t)
                dlog.debug(f"Prepared config {name}")
                name_no_ext = os.path.splitext(name)[0]
                config_results_path = \
                    os.path.join(package_cache_path, "results", name_no_ext)
                os.makedirs(os.path.join(config_results_path, "STATE"))

    def _append_queue(self, package_basename):
        """For convenience, so the user doesn't need to copy/paste the package
        names each time they submit a job. This is a LIFO (last in first out)
        queue, so the most recent primed job will be the one submitted if the
        user does not specify a package."""

        if os.path.exists(self.queue_path):
            queue = yaml.safe_load(open(self.queue_path))
            queue['trials'].append({
                'date_primed': self.dt_string,
                'package_basename': package_basename
            })
        else:
            queue = {
                'trials': [{
                    'date_primed': self.dt_string,
                    'package_basename': package_basename
                }]
            }

        with open(self.queue_path, 'w') as f:
            yaml.dump(queue, f, default_flow_style=False)

    def scaffold(self):
        """Primes the computation by constructing all permutations of jobs
        necessary to be run, and saving this as a single pickle file to disk.
        This file is then read by the RANK=0 MPI process and distributed to
        the workers during execution."""

        package_cache_path, package_basename = self._setup_cache_target()
        configs = self._get_config_files()
        self._ready_package(configs, package_cache_path)
        self._append_queue(package_basename)


class Submitter(BaseOverlord):
    """TODO: detailed docstring"""

    def __init__(self, cl_args):
        super().__init__(cl_args)

    def _load_last_package_from_LIFO_queue(self):
        """Loads the last entry in the saved LIFO queue, returns that entry,
        and pops that entry from the list, resaving the file."""

        queue = yaml.safe_load(open(self.queue_path))

        if len(queue['trials']) < 1:
            msg = "LIFO queue is empty, run prime before execute. Exiting."
            dlog.error(msg)
            exit(0)

        # Get the last entry
        last_entry = queue['trials'][-1]

        # Resave the file
        queue['trials'] = queue['trials'][:-1]
        with open(self.queue_path, 'w') as f:
            yaml.dump(queue, f, default_flow_style=False)

        return last_entry['package_basename']

    def submit_mpi(self):
        """Submits the packages as specified by the cl_args."""

        # Get the absolute path to the package, and raise a critical error if
        # the user-specified package does not exist. We also check the LIFO
        # queue if the package is not specified.
        if self.cl_args.package is not None:
            basename = self.cl_args.package
        else:
            basename = self._load_last_package_from_LIFO_queue()
            bold_basename = utils.bold(basename)
            dlog.info(f"Using package {bold_basename} from LIFO queue")

        package = os.path.join(self.cache_dir, basename)
        if not os.path.isdir(package):
            msg = f"Cached package path {package} does not exist"
            dlog.critical(msg)
            raise RuntimeError(msg)

        # Initialize the SlurmWriter, which checks a default config file
        # specified in the command line arguments, and overrides those defaults
        # with other command line arguments.
        slurm_writer = SlurmWriter(self.cl_args)

        dlog.debug(f"Package path {package}")
        submit_script = os.path.join(package, "submit.sbatch")

        # Write the slurm script regardless of --bash, why not
        slurm_writer.write(submit_script, stream_name=basename)

        dryrun = int(self.cl_args.dryrun)
        debug = int(self.cl_args.debug)
        solver = int(self.cl_args.solver)

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

            local_threads = 1

            exe = f"mpiexec -np {local_procs} python3 ._submit.py " \
                f"{package} {debug} {dryrun} {solver}"
            fname = f"submit_{basename}.sh"
            with open(fname, 'w') as f:
                f.write(f"export OMP_NUM_THREADS={local_threads}\n")
                f.write(exe)
            bash_execute = utils.bold(f"bash {fname}")
            dlog.info(f"Run {bash_execute} to execute trials")

        # Else we go through the protocol of submitting a job via SLURM
        else:
            os.makedirs(utils.JOB_DATA_PATH, exist_ok=True)
            utils.run_command(f"mv {submit_script} .")
            args = f"{package} {debug} {dryrun} {solver}"
            out = utils.run_command(f"sbatch submit.sbatch {args}")
            if out == 0:
                dlog.info(f"{basename} submit - success")
            else:
                dlog.error(f"{basename} submit - failure (err code {out})")
            utils.run_command(f"mv submit.sbatch {package}")
