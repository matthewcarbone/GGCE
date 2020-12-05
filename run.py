#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import logging
import os
import shutil
import sys

from ggce.utils import parser, utils
from ggce.overlord import Prime, Submitter


def purge(file_or_directory):
    """Removes a directory tree or file."""

    if os.path.isdir(file_or_directory):
        shutil.rmtree(file_or_directory)
    elif os.path.exists(file_or_directory):
        os.remove(file_or_directory)


if __name__ == '__main__':

    args = parser.global_parser(sys.argv[1:])

    if args.purge:
        print(f"Removing {utils.get_cache_dir()}")
        print(f"Removing {utils.LIFO_QUEUE_PATH}")
        print(f"Removing {utils.JOB_DATA_PATH}")
        purge(utils.get_cache_dir())
        purge(utils.LIFO_QUEUE_PATH)
        purge(utils.JOB_DATA_PATH)

        # Get the local submit files
        local_sub = os.listdir(".")
        local_sub = [xx for xx in local_sub if "submit_" in xx and ".sh" in xx]
        for file in local_sub:
            print(f"Removing {file}")
            purge(file)
        exit(0)

    # If Debug is false
    if args.debug == 0:
        logging.disable(10)

    if args.protocol == 'prime':
        primer = Prime(args)
        primer.scaffold()

    elif args.protocol == 'execute':
        ex = Submitter(args)
        ex.submit_mpi()

    else:
        raise RuntimeError(f"Unknown protocol {args.protocol}")
