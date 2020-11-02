#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import logging
import numpy as np
import sys
import yaml

from ggce.executor import parallel
from ggce.structures import InputParameters


if __name__ == '__main__':
    nprocs = int(sys.argv[1])
    core_path = str(sys.argv[2])
    debug = int(sys.argv[3])
    w_bins = int(sys.argv[4])

    system_input = yaml.safe_load(open(f"{core_path}/config.yaml"))
    config = InputParameters(**system_input)
    config.init_terms()

    w_grid = np.loadtxt(f"{core_path}/grid.txt")

    # Suppress debug stream
    if debug == 1:
        logging.disable(-1)

    parallel(
        w_grid, config, w_bins=w_bins, nprocs=nprocs, log_every=10,
        target_dir=core_path
    )
