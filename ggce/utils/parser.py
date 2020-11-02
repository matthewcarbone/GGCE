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
        '-p', '--package', type=int, default=None, dest='package',
        help='Index of the package to prime. If None, primes all available '
        'packages. If a single number, primes only that package. Note that '
        'packages must be in the $GMA_PACKAGE_DIR directory (which if not set '
        'defaults to `packages` in the working directory), and must start '
        'with three digits, e.g., 123_my_package_sub_dir. A single package '
        'should contain multiple ##_config.yaml files, a single slurm.yaml '
        'config file, and a slurm_config_mapping.yaml config file.'
    )
    prime_sp.add_argument(
        '--distribute', type=int, default=1, dest='distribute',
        help='Number of nodes to distribute each job onto.'
    )
    prime_sp.add_argument(
        '--procs', type=int, default=1, dest='mp_procs',
        help='The number of multiprocessing processes to use. Note that this '
        'is not the number of MPI processes as set in SLURM, that is always '
        '1. This is the number of multiprocessing jobs to use, each of which '
        'will utilize OMP_NUM_THREADS threads. So the total number of CPUs '
        'should approximately equal procs * OMP_NUM_THREADS.'
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
        help='Number of bosons.'
    )
    req.add_argument(
        '-M', type=int, nargs='+', default=None, dest='M',
        help='Maximal cloud extent.'
    )
    req.add_argument(
        '--eta', type=float, nargs='+', default=None, dest='eta',
        help='Broadening parameter.'
    )

    # (2) ---------------------------------------------------------------------
    execute_sp = subparsers.add_parser(
        "execute", formatter_class=SortingHelpFormatter,
        description='Submits jobs to the job controller or runs them locally '
        'depending on the prime step.'
    )

    execute_sp.add_argument(
        '--debug', default=False, dest='debug', action='store_true',
        help='Enables the debug logging stream to stdout.'
    )

    execute_sp.add_argument(
        '--wbins', type=int, default=-1, dest='w_bins',
        help='Number of approximately-equal length bins to split up the '
        'w-grid into.'
    )

    prime_sp.add_argument(
        '-p', '--package', type=int, nargs='+', default=None, dest='package',
        help='Index of the packages to run. These are indexed by their '
        'numbers in the cache directory. Default is to run all available.'
    )

    # Quick post processing on the value for beta_critical
    args = ap.parse_args(sys_argv)

    if args.protocol == 'prime':
        if (args.N is None or args.M is None or args.eta is None):
            raise RuntimeError("M, N, and eta are required in prime step")

    return args
