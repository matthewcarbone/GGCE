#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import sys

from ggce.utils import parser
from ggce.executor import Primer, Submitter


if __name__ == '__main__':

    args = parser.global_parser(sys.argv[1:])

    if args.protocol == 'prime':
        primer = Primer(args)
        primer.prime()

    elif args.protocol == 'execute':
        ex = Submitter(args)
        ex.run()
