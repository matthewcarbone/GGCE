#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import copy
import logging
import numpy as np
import os
import shlex
import subprocess
import time
import yaml


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
        'mail_type': lambda s: f"#SBATCH --mail-type={s}"
    }

    # Maps basically everything else to the proper format
    OTHERMAP = {
        'threads': lambda ii=None:
            f"export OMP_NUM_THREADS={ii}" if ii is not None else
            "export OMP_NUM_THREADS=1",
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
            f.write("\n")
            for line in other_lines:
                f.write(f"{line}\n")
            f.write(f"\n{last_line}")


class DisableLogger:

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


# https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def N_M_eta_k_subdir(M, N, eta, k_u_pi, mapping, config_idx):
    subdir = f"{mapping['M'][M]:03}/{mapping['N'][N]:03}/"
    subdir += f"{mapping['eta'][eta]:03}/{mapping['k_units_pi'][k_u_pi]:06}"
    return f"{config_idx:03}/{subdir}"


def bold(s):
    """Makes a string bold for console output."""

    return Color.BOLD + s + Color.END


def get_cache_dir():
    cache = os.environ.get('GGCE_CACHE_DIR')
    if cache is None:
        cache = 'results'
    os.makedirs(cache, exist_ok=True)
    dlog.debug(f"Cache directory set to {cache}")
    return cache


def get_package_dir():
    cache = os.environ.get('GGCE_PACKAGES_DIR')
    if cache is None:
        cache = 'packages'
    os.makedirs(cache, exist_ok=True)
    dlog.debug(f"Package directory set to {cache}")
    return cache


def listdir_fullpath(d):
    """https://stackoverflow.com/a/120948"""

    return [os.path.join(d, f) for f in os.listdir(d)]


def listdir_files_fp(d):
    x = [os.path.join(d, f) for f in os.listdir(d)]
    return [xx for xx in x if not os.path.isdir(xx)]


def listdir_fullpath_dirs_only(d):
    dirs = [os.path.join(d, f) for f in os.listdir(d)]
    return [d for d in dirs if os.path.isdir(d)]


def time_func(arg1=None):
    """source: http://scottlobdell.me/2015/04/decorators-arguments-python/"""

    def real_decorator(function):

        def wrapper(*args, **kwargs):

            aa = arg1
            if aa is None:
                aa = function.__name__

            t1 = time.time()
            x = function(*args, **kwargs)
            t2 = time.time()
            elapsed = (t2 - t1) / 60.0
            print(f"\t{aa} done {elapsed:.02f} m")
            return x

        return wrapper

    return real_decorator


def configuration_space_generator(length, total_sum):
    """Generator for yielding all possible combinations of integers of length
    `length` that sum to total_sum. Not that cases such as length = 4 and
    total_sum = 5 like [0, 0, 2, 3] need to be screened out, since these do
    not correspond to valid f-functions.

    Source of algorithm:
    https://stackoverflow.com/questions/7748442/
    generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """

    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in configuration_space_generator(
                length - 1, total_sum - value
            ):
                r = (value,) + permutation
                yield r


class ConfigFilter:

    def __init__(self, M_min=2, filter_type=None):
        """M_min is the minimum M in which we want the filter to apply."""

        self.M_min = M_min
        self.filter_type = filter_type
        self.config_bound_map = dict()

    def _find(self, M, N):
        """Attempts to find if a rule has been calculated already for some
        configuration. Returns the configuration if it does, else returns
        None."""

        try:
            d1 = self.config_bound_map[M]
        except KeyError:
            self.config_bound_map[M] = dict()
            return None

        try:
            d2 = d1[N]
        except KeyError:
            return None

        return d2

    def get(self, M, N):
        """Get's the maximum configuration allowed by the rule."""

        possible = self._find(M, N)
        if possible is not None:
            return possible

        if M < self.M_min:
            ans = np.array([N for _ in range(M)])

        elif self.filter_type == 'gaussian':
            dev = M**2 / 4.0 / np.log(N)
            x = np.array([(-(M / 2) + ii + 0.5) for ii in range(M)])
            ans = (N * np.exp(-x**2 / dev)).astype(int)

        else:
            raise NotImplementedError(f"Filter {self.config_filter}")

        self.config_bound_map[M][N] = ans
        return ans

    def visualize(self, M, N):
        n = self.get(M, N)
        print("\n")
        for nn in n:
            print(f"{nn} \t | ", end="")
            for mm in range(0, nn):
                print("#", end="")
            print("\n", end="")

    def __call__(self, n):

        M = len(n)

        # G is always legal
        if M == 1 and n[0] == 0:
            return True

        # The case in which either of the edges is zero is always illegal,
        # except for G
        if n[0] <= 0 or n[-1] <= 0:
            return False

        # No filter, return True always
        if self.filter_type is None:
            return True

        # Then check the rule
        N = sum(n)

        # Upper bound
        bound = self.get(M, N)
        if np.any((bound - np.array(n)) < 0):
            return False
        return True


def time_remaining(time_elapsed, percentage_complete):
    """Returns the time remaining."""

    # time_elapsed / percent_elapsed = time_remaining / pc_remaining
    # time_remaining = time_elapased / percent_elapsed * pc_remaining
    if percentage_complete == 100:
        return 0.0
    return (100.0 - percentage_complete) * time_elapsed / percentage_complete


def run_command(command, silent=True):
    """https://www.endpoint.com/blog/2015/01/28/
    getting-realtime-output-using-python"""

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output and not silent:
            dlog.debug(output.strip().decode())

    rc = process.poll()
    return rc
