#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from operator import attrgetter


# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(ArgumentDefaultsHelpFormatter, HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def global_parser(sys_argv):

    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)

    ap.add_argument(
        '--force', dest='force', default=False, action='store_true',
        help='Overrides failsafes, automatically applies "yes" to otherwise '
        'required user inputs.'
    )

    ap.add_argument(
        '--debug', default=False, dest='debug', action='store_true',
        help='Enables the debug logging stream to stdout.'
    )

    subparsers = ap.add_subparsers(
        help='Choices for various priming, execution and post-processing '
        'protocols.', dest='protocol'
    )

    # (1) ---------------------------------------------------------------------
    prime_sp = subparsers.add_parser(
        "prime", formatter_class=SortingHelpFormatter,
        description='Prime the computation for submission by creating the '
        'appropriate directories and writing the SLURM submit file.'
    )
    prime_sp.add_argument(
        '-c', '--configs', type=int, nargs='+', default=None, dest='c_to_run',
        help='Indexes the configs within a package to run. Defaults to '
        'running all configs in the package.'
    )
    prime_sp.add_argument(
        '--linspacek', default=False, dest='linspacek',
        action='store_true',
        help='If True, then the user is required to provide three values for '
        'the -k flag: the k0, kf and total number of k points.'
    )

    # Local stuff can be done easily via Jupyter notebooks
    # prime_sp.add_argument(
    #     '--local', dest='local', default=False, action='store_true',
    #     help='Saves a bash script instead of SLURM.'
    # )
    # prime_sp.add_argument(
    #     '-r', '--rule', dest='rule', default=0, type=int,
    #     help="Special rules for further restricting the configuration space."
    #     "Default is 0 (no rule)."
    # )

    req = prime_sp.add_argument_group("required")
    req.add_argument(
        '-N', type=int, nargs='+', default=None, dest='N',
        help='Number of bosons.', required=True
    )
    req.add_argument(
        '-M', type=int, nargs='+', default=None, dest='M',
        help='Maximal cloud extent.', required=True
    )
    req.add_argument(
        '--eta', type=float, nargs='+', default=None, dest='eta',
        help='Broadening parameter.', required=True
    )
    req.add_argument(
        '-k', type=float, nargs='+', default=None, dest='k_units_pi',
        help='Values for k in units of pi.', required=True
    )
    req.add_argument(
        '-p', '--package', type=int, default=None, dest='package',
        help='Index of the package to prime. If None, primes all available '
        'packages. If a single number, primes only that package. Note that '
        'packages must be in the $GMA_PACKAGE_DIR directory (which if not set '
        'defaults to `packages` in the working directory), and must start '
        'with three digits, e.g., 123_my_package_sub_dir. A single package '
        'should contain multiple ##_config.yaml files, a single slurm.yaml '
        'config file, and a slurm_config_mapping.yaml config file.',
        required=True
    )

    # (2) ---------------------------------------------------------------------
    execute_sp = subparsers.add_parser(
        "execute", formatter_class=SortingHelpFormatter,
        description='Submits jobs to the job controller or runs them locally '
        'depending on the prime step.'
    )
    execute_sp.add_argument(
        '--bash', default=False, dest='bash',
        action='store_true', help='Run using bash (locally) instead of sbatch'
    )
    execute_sp.add_argument(
        '--dryrun', default=False, dest='dryrun',
        action='store_true', help='Run in dryrun mode. This is different '
        'from debug in the sense that an actual job will be run/submitted, '
        'but all values for G will be randomly sampled, and there will be '
        'no actual GGCE calculations run.'
    )

    slurm = execute_sp.add_argument_group(
        "SLURM", "SLURM script parameters used to override defaults "
        "in the slurm_config.yaml file. Note that some parameters must "
        "be set in the config."
    )
    slurm.add_argument(
        '--config_path', dest='loaded_config_path',
        default='slurm_config.yaml', type=str, help="SLURM config path"
    )
    slurm.add_argument('-N', '--nodes', dest='nodes', default=None, type=int)
    slurm.add_argument(
        '-s', '--tasks_per_node', dest='tasks_per_node', default=None,
        type=int
    )
    slurm.add_argument(
        '-q', '--queue', dest='queue', default=None, type=str
    )
    slurm.add_argument(
        '-p', '--partition', dest='partition', default=None, type=str
    )
    slurm.add_argument('-m', '--mem', dest='memory', default=None, type=str)
    slurm.add_argument(
        '-d', '--threads', default=1, dest='threads', type=int,
        help='Number of threads PER PROCESS.'
    )
    slurm.add_argument('-t', '--time', default=None, dest='time', type=str)

    req = execute_sp.add_argument_group("required")
    req.add_argument(
        '-P', '--package', type=int, nargs='+', default=None, dest='package',
        help='Index of the packages to run. These are indexed by their '
        'numbers in the cache directory.', required=True
    )

    # Quick post processing on the value for beta_critical
    args = ap.parse_args(sys_argv)

    return args
