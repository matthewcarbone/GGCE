#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import logging
import sys

from ggce.utils import parser
from ggce.overlord import Prime, Submitter


if __name__ == '__main__':

    args = parser.global_parser(sys.argv[1:])

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
