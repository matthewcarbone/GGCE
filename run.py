#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

# test edit

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
        print(f"Removed tree {file_or_directory}")
    elif os.path.exists(file_or_directory):
        os.remove(file_or_directory)
        print(f"Removed file {file_or_directory}")


if __name__ == '__main__':

    args = parser.global_parser(sys.argv[1:])

    if args.purge:
        purge(utils.get_cache_dir())
        purge(utils.LIFO_QUEUE_PATH)
        purge(utils.JOB_DATA_PATH)

        # Get the local submit files
        local_sub = os.listdir(".")
        local_sub = [xx for xx in local_sub if "submit_" in xx and ".sh" in xx]
        for file in local_sub:
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
